import torch
import random, os
import numpy as np
from minatar import Environment
from utils.Agent_Nets import Q_ConvNet
from tensorboardX import SummaryWriter
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='MinAtar')
parser.add_argument('--id', type=str, default='DQN', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--game', type=str, default='asterix', help='Game')
parser.add_argument('--use-cuda', type=str, default='True', help='Disable CUDA')
parser.add_argument('--Eval-episode', type=int, default=int(30), metavar='Episodes', help='Number of evaluating episode')
parser.add_argument('--T-max', type=int, default=int(5e6), metavar='STEPS', help='Number of training steps')
parser.add_argument('--dueling', type=str, default='False', help='Dueling Network Architecture')
parser.add_argument('--double', type=str, default='False', help='Double DQN')
parser.add_argument('--n-step', type=int, default=1, help='Multi-step DQN')
parser.add_argument('--distributional', type=str, default='False', help='Distributional DQN')
parser.add_argument('--noisy', type=str, default='False', help='Noisy DQN')



class Eval_Agent():
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
        else:
            args.device = torch.device('cpu')

        self.save_model_dir = os.path.join('results/alg={}/env={},seed={}/saved_models'.format(args.id, args.game, args.seed))
        if not os.path.exists('results/alg={}/env={},seed={}'.format(args.id, args.game, args.seed)):
            print("cannot find {}".format('results/alg={}/env={},seed={}'.format(args.id, args.game, args.seed)))
            exit()
            os.makedirs('results/alg={}/env={},seed={}/eval_logs')
        self.writer = SummaryWriter('results/alg={}/env={},seed={}/eval_logs'.format(args.id, args.game, args.seed))

        self.eval_env = Environment(args.game)
        self.obs_dim = self.eval_env.state_shape()[2]
        self.act_dim = self.eval_env.num_actions()

        self.QValue_Net = Q_ConvNet(in_channels=self.obs_dim,
                            num_actions=self.act_dim,
                            dueling=args.dueling,
                            noisy=args.noisy,
                            distributional=args.distributional,
                            atom_size=51,
                            v_min=-10.0,
                            v_max=10.0).to(args.device)
        self.QValue_Net.eval()


    def get_state(self, s):
        return (torch.tensor(s, device=args.device).permute(2, 0, 1)).unsqueeze(0).float()

    def Eval(self):

        for step in tqdm(range(int(5e4), self.args.T_max + 1, int(5e4))):
            if not os.path.exists(self.save_model_dir + '/training_step={}_checkpoint.pth'.format(step)):
                print("cannot find model {}".format(self.save_model_dir + '/training_step={}_checkpoint.pth'.format(step)))
                exit()

            self.QValue_Net.load_state_dict(torch.load(self.save_model_dir + '/training_step={}_checkpoint.pth'.format(step)).state_dict())
            print("eval model learning step={}...\n".format(step))
            score = 0.0
            for episode in tqdm(range(self.args.Eval_episode)):
                self.eval_env.reset()
                state = self.get_state(self.eval_env.state())
                done = False
                while (not done):
                    next_state, action, reward, done = self.Eval_Interaction(state, self.eval_env)
                    score += reward.item()
                    state = next_state
            R = score / self.args.Eval_episode
            self.writer.add_scalar("Eval/RETURN/STEP", R, global_step=step)
            print("Return={}".format(R))
        self.writer.close()

    def Eval_Interaction(self, state, eval_env):
        with torch.no_grad():
            action = self.QValue_Net(state).max(1)[1].view(1, 1)
        reward, done = eval_env.act(action)
        next_state = self.get_state(eval_env.state())
        return next_state, action, torch.tensor([[reward]], device=self.args.device).float(), torch.tensor([[done]], device=self.args.device)

def Eval_after_Train(args):
    Agent = Eval_Agent(args)
    Agent.Eval()


if __name__ == '__main__':
    args = parser.parse_args()
    Agent = Eval_Agent(args)
    Agent.Eval()