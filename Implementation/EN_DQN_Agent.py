import torch
import torch.nn.functional as f
import torch.optim as optim
import random, logging, os
import numpy as np
from collections import namedtuple, deque
from minatar import Environment
from utils.Replay_Buffer import replay_buffer
from utils.Agent_Nets import MultiHead_Q_ConvNet, MultiHead_Q_Bone, MultiHead_Q_Head
from tensorboardX import SummaryWriter
import argparse
from tqdm import tqdm
import time
# from Run_Eval import Eval_after_Train

transition = namedtuple('transition', 'state, action, reward, next_state, done')
parser = argparse.ArgumentParser(description='MinAtar')
parser.add_argument('--id', type=str, default='EnsembleDQN', help='Experiment ID')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--game', type=str, default='seaquest', help='Game')
parser.add_argument('--use-cuda', type=bool, default=True, help='Disable CUDA')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--target-update', type=int, default=int(1e3), metavar='STEPS', help='Number of steps after which to update target network')
parser.add_argument('--T-max', type=int, default=int(5e6), metavar='STEPS', help='Number of training steps')
parser.add_argument('--first-n-frames', type=int, default=int(1e5), metavar='STEPS', help='Number of random')
parser.add_argument('--learn-start', type=int, default=int(5e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--start-epsilon', type=float, default=1, metavar='ε', help='Start Epsilon')
parser.add_argument('--end-epsilon', type=float, default=0.01, metavar='ε', help='End Epsilon')
parser.add_argument('--learning-rate', type=float, default=0.00025, metavar='$/alpha$', help='Learning rate')
parser.add_argument('--grad-momentum', type=float, default=0.95, metavar='η', help='Adam')
parser.add_argument('--squared-grad-momentum', type=float, default=0.95, metavar='η', help='Adam')
parser.add_argument('--min-squared-grad', type=float, default=0.01, metavar='η', help='Adam')
parser.add_argument('--gamma', type=float, default=0.99, metavar='γ', help='Discount factor')

parser.add_argument('--ensemble-size', type=int, default=5, help='Ensemble size')

parser.add_argument('--Eval', type=bool, default=False, help='Eval after train finished')
parser.add_argument('--Eval_episode', type=int, default=int(30), help='Evaluation Episode')

class DQN_Agent():
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available() and args.use_cuda:
            args.device = torch.device('cuda')
            torch.cuda.manual_seed(np.random.randint(1, 10000))
        else:
            args.device = torch.device('cpu')
        self.results_dir = 'results/alg={},ensemble_size={}/env={},seed={}'.format(args.id, args.ensemble_size, args.game, args.seed)
        os.makedirs(self.results_dir + '/saved_models')
        os.makedirs(self.results_dir + '/logs')
        self.writer = SummaryWriter(self.results_dir + '/logs')
        print(' ' * 26 + 'Options')
        self.options_file = open(self.results_dir + "/options.txt", "w")
        for k, v in vars(args).items():
            print(' ' * 26 + k + ': ' + str(v))
            self.options_file.write(k + ': ' + str(v) + '\n')
        self.options_file.write("start time" + ': ' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '\n')
        self.act_env = Environment(args.game)
        self.obs_dim = self.act_env.state_shape()[2]
        self.act_dim = self.act_env.num_actions()

        self.QValue_Net_Bone = MultiHead_Q_Bone(in_channels=self.obs_dim).to(args.device)
        self.Bone_Optimizer = optim.RMSprop(self.QValue_Net_Bone.parameters(),
                                            lr=args.learning_rate,
                                            alpha=args.squared_grad_momentum,
                                            centered=True,
                                            eps=args.min_squared_grad)
        self.Target_Net_Bone = MultiHead_Q_Bone(in_channels=self.obs_dim).to(args.device)
        self.Target_Net_Bone.load_state_dict(self.QValue_Net_Bone.state_dict())
        self.Target_Net_Bone.eval()

        self.QValue_Net_Heads = []
        self.Target_Net_Heads = []
        self.Heads_Optimizer = []
        for head in range(self.args.ensemble_size):
            self.QValue_Net_Heads.append(MultiHead_Q_Head(num_actions=self.act_dim).to(args.device))
            self.Heads_Optimizer.append(optim.RMSprop(self.QValue_Net_Heads[head].parameters(),
                                                      lr=args.learning_rate,
                                                      alpha=args.squared_grad_momentum,
                                                      centered=True,
                                                      eps=args.min_squared_grad))
            self.Target_Net_Heads.append(MultiHead_Q_Head(num_actions=self.act_dim).to(args.device))
            self.Target_Net_Heads[head].load_state_dict(self.QValue_Net_Heads[head].state_dict())
            self.Target_Net_Heads[head].eval()

        self.memory = replay_buffer(args.memory_capacity, self.get_state(self.act_env.state()).size(), args)

    def get_state(self, s):
        return (torch.tensor(s, device=args.device).permute(2, 0, 1)).unsqueeze(0).float()

    def Train(self):
        train_start_time = time.time()
        avg_return = 0.0
        step = 0
        episode = 0
        Net_update_counter = 0

        while step < self.args.T_max:
            score = 0.0
            self.act_env.reset()
            state = self.get_state(self.act_env.state())
            done = False
            while (not done) and step < self.args.T_max:
                next_state, action, reward, done = self.Interaction(step, state, self.act_env)
                self.memory.store(state.to(self.args.device),
                                  action.unsqueeze(dim=0).to(self.args.device),
                                  reward,
                                  next_state.to(self.args.device),
                                  done)
                if step > self.args.learn_start and self.memory.buffer_len >= self.args.batch_size:
                    sample = self.memory.sample(self.args.batch_size)
                    trained_head = np.random.randint(self.args.ensemble_size)
                    Q_values = self.Learning(sample, trained_head)
                    Net_update_counter += 1
                # Update the target network
                if Net_update_counter > 0 and Net_update_counter % self.args.target_update == 0:
                    self.Target_Net_Bone.load_state_dict(self.QValue_Net_Bone.state_dict())
                    for head in range(self.args.ensemble_size):
                        self.Target_Net_Heads[head].load_state_dict(self.QValue_Net_Heads[head].state_dict())
                score += reward.item()
                step += 1
                state = next_state
                if step % 10000 == 0:
                    train_now_time = time.time()
                    self.writer.add_scalar("Train/AVR_RETURN/STEP", avg_return, global_step=step)
                    self.writer.add_scalar("Train/Q_VALUE/STEP", Q_values, global_step=step)
                    print("[{}/{}]:[avg_return={},q_values={}]-estimate-{}min".format(step, self.args.T_max, avg_return, Q_values, time.asctime(time.localtime(time.time()+(train_now_time-train_start_time)*(self.args.T_max/step-1)))))
                # if step % 50000 == 0:
                    # torch.save(self.QValue_Net, self.results_dir + '/saved_models/training_step={}_bone_checkpoint.pth'.format(step))
                    # print("[{}/{}]save checkpoint at {}".format(step, self.args.T_max, self.results_dir + '/saved_models'))
            episode += 1
            avg_return = 0.99 * avg_return + 0.01 * score
        self.writer.close()
        self.options_file.write("end time" + ': ' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '\n')
        self.options_file.close()
        # if self.args.Eval == True:
        #     Eval_after_Train(self.args)

    def Interaction(self, step, state, act_env):

        if step < self.args.learn_start:
            action = torch.tensor([[random.randrange(self.act_dim)]], device=self.args.device)
        else:
            if step - self.args.learn_start >= self.args.first_n_frames:
                epsilon = self.args.end_epsilon
            else:
                epsilon = ((self.args.end_epsilon - self.args.start_epsilon) / self.args.first_n_frames) * (step - self.args.learn_start) + self.args.start_epsilon
            # epsilon greedy
            if np.random.binomial(1, epsilon) == 1:
                action = torch.tensor([[random.randrange(self.act_dim)]], device=self.args.device)
            else:
                with torch.no_grad():
                    x = self.QValue_Net_Bone(state)
                    for head in range(self.args.ensemble_size):
                        if head == 0:
                            q_values = self.QValue_Net_Heads[head](x)
                        else:
                            q_values += self.QValue_Net_Heads[head](x)
                    action = q_values.max(1)[1].view(1, 1)
        reward, done = act_env.act(action)
        next_state = self.get_state(act_env.state())
        return next_state, action, torch.tensor([[reward]], device=self.args.device).float(), torch.tensor([[done]], device=self.args.device)

    def Learning(self, sample, trained_head):
        states, actions, rewards, next_states, dones = sample
        states = states.squeeze(dim=1)
        actions = actions.squeeze(dim=1).long()
        rewards = rewards.squeeze(dim=1)
        next_states = next_states.squeeze(dim=1)
        dones = dones.squeeze(dim=1).bool()

        x = self.QValue_Net_Bone(states)
        Q_s_a = self.QValue_Net_Heads[trained_head](x).gather(1, actions)

        with torch.no_grad():
            none_done_next_state_index = torch.tensor([i for i, is_term in enumerate(dones) if is_term == 0], dtype=torch.int64, device=self.args.device)
            none_done_next_states = next_states.index_select(0, none_done_next_state_index)
            Q_s_prime_a_prime = torch.zeros(self.args.batch_size, 1, device=self.args.device)
            if len(none_done_next_states) != 0:
                x = self.Target_Net_Bone(none_done_next_states)
                for head in range(self.args.ensemble_size):
                    if head == 0:
                        q_values = self.Target_Net_Heads[head](x) / self.args.ensemble_size
                    else:
                        q_values += self.Target_Net_Heads[head](x) / self.args.ensemble_size
                Q_s_prime_a_prime[none_done_next_state_index] = q_values.detach().max(1)[0].unsqueeze(1)
            target = rewards + self.args.gamma * Q_s_prime_a_prime

        # Huber loss
        loss = f.smooth_l1_loss(target, Q_s_a)
        # Zero gradients, backprop, update the weights of policy_net
        self.Bone_Optimizer.zero_grad()
        self.Heads_Optimizer[trained_head].zero_grad()
        loss.backward()
        self.Bone_Optimizer.step()
        self.Heads_Optimizer[trained_head].step()
        return Q_s_a.mean().detach().cpu().numpy()

if __name__ == '__main__':
    args = parser.parse_args()
    for args.seed in [3]:
        Agent = DQN_Agent(args)
        Agent.Train()
