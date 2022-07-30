import torch
import torch.nn.functional as f
import torch.optim as optim
import random, logging, os
import numpy as np
from collections import namedtuple, deque
from minatar import Environment
from utils.Replay_Buffer import retrace_replay_buffer
from utils.Agent_Nets import Q_ConvNet
from tensorboardX import SummaryWriter
import argparse
from tqdm import tqdm
import time
from run_evaluation import Eval_after_Train


transition = namedtuple('transition', 'state, action, reward, next_state, done')
parser = argparse.ArgumentParser(description='MinAtar')
parser.add_argument('--id', type=str, default='Retrace', help='Experiment ID')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--game', type=str, default='asterix', help='Game')
parser.add_argument('--use-cuda', type=bool, default=True, help='Disable CUDA')
parser.add_argument('--batch-size', type=int, default=1, metavar='SIZE', help='Batch size')
parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--target-update', type=int, default=int(1e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--T-max', type=int, default=int(5e6), metavar='STEPS', help='Number of training steps')
parser.add_argument('--first-n-frames', type=int, default=int(1e5), metavar='STEPS', help='Number of random')
parser.add_argument('--learn-start', type=int, default=int(5e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--start-epsilon', type=float, default=1, metavar='STEPS', help='Start Epsilon')
parser.add_argument('--end-epsilon', type=float, default=0.1, metavar='STEPS', help='End Epsilon')
parser.add_argument('--learning-rate', type=float, default=0.00025, metavar='η', help='Learning rate')
parser.add_argument('--grad-momentum', type=float, default=0.95, metavar='η', help='Adam')
parser.add_argument('--squared-grad-momentum', type=float, default=0.95, metavar='η', help='Adam')
parser.add_argument('--min-squared-grad', type=float, default=0.01, metavar='η', help='Adam')
parser.add_argument('--gamma', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--p_lambda', type=float, default=1, metavar='lambda', help='Lambda factor')

class Retrace_Agent():
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
        self.results_dir = os.path.join('results/alg={}/env={},seed={}'.format(args.id, args.game, args.seed))
        if not os.path.exists(self.results_dir):
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
        self.QValue_Net = Q_ConvNet(in_channels=self.obs_dim,
                            num_actions=self.act_dim,
                            dueling=False,
                            noisy=False,
                            distributional=False,
                            atom_size=51,
                            v_min=-10.0,
                            v_max=10.0).to(args.device)

        self.Target_Net = Q_ConvNet(in_channels=self.obs_dim,
                            num_actions=self.act_dim,
                            dueling=False,
                            noisy=False,
                            distributional=False,
                            atom_size=51,
                            v_min=-10.0,
                            v_max=10.0).to(args.device)

        self.Target_Net.load_state_dict(self.QValue_Net.state_dict())
        self.Target_Net.eval()
        self.Optimizer = optim.RMSprop(self.QValue_Net.parameters(), lr=args.learning_rate, alpha=args.squared_grad_momentum, centered=True, eps=args.min_squared_grad)
        self.memory = retrace_replay_buffer(args.memory_capacity)
    def get_state(self, s):
        return (torch.tensor(s, device=args.device).permute(2, 0, 1)).unsqueeze(0).float()
    def Train(self):
        train_start_time = time.time()
        avg_return = 0.0
        step = 0
        episode = 0
        QValueNet_update_counter = 0
        while step < self.args.T_max:
            score = 0.0
            self.act_env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_behaviour_probs = []
            episode_target_probs = []

            state = self.get_state(self.act_env.state())
            episode_states.append(state)
            done = False
            while (not done) and step < self.args.T_max:
                # collect data
                next_state, action, reward, done, behaviour_probs, target_probs = self.Interaction(step, state, self.act_env)
                episode_states.append(next_state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_dones.append(done)
                episode_behaviour_probs.append(behaviour_probs)
                episode_target_probs.append(target_probs)

                # sample data
                if step > self.args.learn_start and len(self.memory.buffer) >= self.args.batch_size:
                    sample = self.memory.sample(self.args.batch_size)
                    Q_values = self.Learning(sample)
                    QValueNet_update_counter += 1
                # Update the target network
                if QValueNet_update_counter > 0 and QValueNet_update_counter % self.args.target_update == 0:
                    self.Target_Net.load_state_dict(self.QValue_Net.state_dict())

                score += reward.item()
                step += 1
                state = next_state
                if step % 10000 == 0:
                    train_now_time = time.time()
                    self.writer.add_scalar("Train/AVR_RETURN/STEP", avg_return, global_step=step)
                    self.writer.add_scalar("Train/Q_VALUE/STEP", Q_values, global_step=step)
                    print("[{}/{}]:[avg_return={},q_values={}]-estimate-{}min".format(step, self.args.T_max, avg_return, Q_values, time.asctime(time.localtime(time.time()+(train_now_time-train_start_time)*(self.args.T_max/step-1)))))
                if step % 50000 == 0:
                    torch.save(self.QValue_Net, self.results_dir + '/saved_models/training_step={}_checkpoint.pth'.format(step))
                    print("[{}/{}]save checkpoint at {}".format(step, self.args.T_max, self.results_dir + '/saved_models'))

            # save traj            
            self.memory.store(
                episode_states,
                episode_actions,
                episode_rewards,
                episode_dones,
                episode_behaviour_probs,
                episode_target_probs)

            episode += 1
            avg_return = 0.99 * avg_return + 0.01 * score
        self.writer.close()
        self.options_file.write("start time" + ': ' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '\n')
        self.options_file.close()
        Eval_after_Train(self.args)


    def Interaction(self, step, state, act_env):

        if step - self.args.learn_start >= self.args.first_n_frames:
            epsilon = self.args.end_epsilon
        else:
            epsilon = ((self.args.end_epsilon - self.args.start_epsilon) / self.args.first_n_frames) * (step - self.args.learn_start) + self.args.start_epsilon

        if step < self.args.learn_start:
            action = torch.tensor([[random.randrange(self.act_dim)]], device=self.args.device)
        else:
            # epsilon greedy
            if np.random.binomial(1, epsilon) == 1:
                action = torch.tensor([[random.randrange(self.act_dim)]], device=self.args.device)
            else:
                with torch.no_grad():
                    action = self.QValue_Net(state).max(1)[1].view(1, 1)

        # calculate behaviour policy probs
        # 1 epsilon part
        epsilon_probs = epsilon * torch.ones(self.act_dim, device=self.args.device) / self.act_dim
        # 2 value function part
        behaviour_probs = epsilon_probs + (1-epsilon) * self.QValue_Net(state) / self.QValue_Net(state).sum()
        behaviour_prob = behaviour_probs.gather(1, action)
        target_probs = epsilon_probs + (1-epsilon) * self.Target_Net(state) / self.Target_Net(state).sum()
        target_prob = behaviour_probs.gather(1, action)


        reward, done = act_env.act(action)
        next_state = self.get_state(act_env.state())

        return next_state, action, torch.tensor([[reward]], device=self.args.device).float(), torch.tensor([[done]], device=self.args.device), behaviour_prob, target_prob

    def Learning(self, sample, weight=None, indices=None):
        episode_states, episode_actions, episode_rewards, episode_dones, episode_behaviour_probs, episode_target_probs = sample
        states = torch.stack(episode_states[:-1]).squeeze(dim=1)
        actions = torch.stack(episode_actions).squeeze(dim=1).long()
        rewards = torch.stack(episode_rewards).squeeze(dim=1)
        next_states = torch.stack(episode_states[1:]).squeeze(dim=1)
        dones = torch.stack(episode_dones).squeeze(dim=1).bool()
        behaviour_probs = torch.stack(episode_behaviour_probs).squeeze(dim=1)
        target_probs = torch.stack(episode_target_probs).squeeze(dim=1)

        # retrace
        C_s = target_probs / behaviour_probs
        C_s = self.args.p_lambda * torch.cat((C_s, torch.ones(C_s.size(), device=self.args.device)), 1).min(1)[0].unsqueeze(1)

        episode_len = C_s.size()[0]

        Q_s_a = self.QValue_Net(states).gather(1, actions)

        with torch.no_grad():
            none_done_next_state_index = torch.tensor([i for i, is_term in enumerate(dones) if is_term == 0],
                                                    dtype=torch.int64, device=self.args.device)
            none_done_next_states = next_states.index_select(0, none_done_next_state_index)
            # expected target value
            Expected_Q_s_prime_a_prime = torch.zeros(Q_s_a.size(), device=self.args.device)
            Q_s_prime_a_prime = torch.zeros(Q_s_a.size(), device=self.args.device)
            if len(none_done_next_states) != 0:
                Expected_Q_s_prime_a_prime[none_done_next_state_index] = self.Target_Net(none_done_next_states).detach().mean(1).unsqueeze(1)
                Q_s_prime_a_prime[none_done_next_state_index] = self.Target_Net(none_done_next_states).detach().max(1)[0].unsqueeze(1)

            target = Q_s_prime_a_prime
            # check every t
            for t in reversed(range(1, episode_len)):
                target[t-1, :] = rewards[t, :] + self.args.gamma * C_s[t, :] * (target[t, :] - Q_s_prime_a_prime[t, :]) + self.args.gamma * Expected_Q_s_prime_a_prime[t, :]
            
        # Huber loss
        loss = f.smooth_l1_loss(target, Q_s_a)
        
        # Zero gradients, backprop, update the weights of policy_net
        self.Optimizer.zero_grad()
        loss.backward()
        self.Optimizer.step()
        return Q_s_a.mean().detach().cpu().numpy()

if __name__ == '__main__':
    args = parser.parse_args()
    Agent = Retrace_Agent(args)
    Agent.Train()
