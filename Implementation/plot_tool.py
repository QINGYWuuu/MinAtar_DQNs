from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import argparse
import tqdm

parser = argparse.ArgumentParser(description='MinAtar')
parser.add_argument('--id', type=str, default='Rainbow', help='Experiment ID')
parser.add_argument('--seed', type=int, default=4, help='Random seed')
parser.add_argument('--game', type=str, default='asterix', help='Game')
parser.add_argument('--use-cuda', type=bool, default=True, help='Disable CUDA')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
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
parser.add_argument('--dueling', type=bool, default=True, help='Dueling Network Architecture')
parser.add_argument('--double', type=bool, default=True, help='Double DQN')
parser.add_argument('--n-step', type=int, default=3, help='Multi-step DQN')
parser.add_argument('--distributional', type=bool, default=True, help='Distributional DQN')
parser.add_argument('--noisy', type=bool, default=True, help='Noisy DQN')
parser.add_argument('--per', type=bool, default=True, help='Periorized Experience Replay')

def main():
    args = parser.parse_args()
    Steps = []
    Values = []
    for args.seed in [4]:
        results_dir = os.path.join('results/alg={}/env={},seed={}/logs'.format(args.id, args.game, args.seed))
        event_data = event_accumulator.EventAccumulator(results_dir)  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        keys = event_data.scalars.Keys()  # get all tags,save in a list
        df = pd.DataFrame(columns=keys)  # my first column is training loss per iteration, so I abandon it
        for key in keys:
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
        Steps.append(np.array(list(range(df[keys[0]].values.size))))
        # Steps.append(np.array(list(range(int(5e4), args.T_max + 1, int(5e4)))))
        Values.append(df[keys[0]].values)
    Steps = np.array(Steps).reshape((-1, 1)).squeeze()
    Values = np.array(Values).reshape((-1, 1)).squeeze()
    sns.lineplot(x=Steps, y=Values, label=args.id)
    plt.title(keys[0], {'size': 15})
    plt.show(bbox_inches="tight", pad_inches=0.0)



main()

#
#
# def plot_data(file_path, alg_name, seeds=range(5),color):
#     Steps = []
#     Values = []
#     for seed in seeds:
#         if os.path.exists(file_path.format(seed)) is False:
#             continue
#         df = pd.read_csv(file_path.format(seed))
#         Steps.append(df.Step)
#         Values.append(df.Value)
#     Steps = np.array(Steps).reshape((-1, 1)).squeeze()
#     Values = np.array(Values).reshape((-1, 1)).squeeze()
#
#     sns.lineplot(x=Steps, y=Values, label=alg_name,color=color)
#
#     for i in range(1):
#         fig, ax = plt.subplots()
#         for alg_id in [0,2,3,5,8,10]:
#
#             plot_data("the_results/MinAtari/{}/run-{}_{}".format(game_name[i], game_name[i], plot_algs[alg_id]) + "_seed={}-tag-AVR_RETURN_STEP.csv",
#                     Alg_Name[alg_id],
#                     range(5),
#                       color=plot_color[alg_id])
#
#         plt.title(Game_Name[i], {'size':15})
#         plt.ticklabel_format(axis='x', style="sci", scilimits=(0, 0))
#         plt.xlabel("Steps", {'size':15})
#         plt.ylabel("Average Return", {'size':15})
#         plt.legend(loc=2, fontsize=10)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         plt.show(bbox_inches="tight", pad_inches=0.0)
#         # plt.savefig("the_curves/MinAtari/{}.pdf".format(Game_Name[i]))
#         plt.close()
