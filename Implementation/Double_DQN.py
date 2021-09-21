import torch
import torch.nn.functional as f
import torch.optim as optim
from minatar import Environment
import random, logging, os
import numpy as np
from collections import namedtuple

from utils.Replay_Buffer import tuple_replay_buffer
from utils.Agent_Nets import Q_ConvNet
from tensorboardX import SummaryWriter

transition = namedtuple('transition', 'state, action, reward, next_state, done')

def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()

def act_env(step, act_dim, state, env, current_net):
    if step < REPLAY_START_SIZE:
        action = torch.tensor([[random.randrange(act_dim)]], device=device)
    else:
        if step - REPLAY_START_SIZE >= FIRST_N_FRAMES:
            epsilon = END_EPSILON
        else:
            epsilon = ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (step - REPLAY_START_SIZE) + EPSILON
        if np.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(act_dim)]], device=device)
        else:
            with torch.no_grad():
                action = current_net(state).max(1)[1].view(1, 1)
    reward, done = env.act(action)
    next_state = get_state(env.state())
    return next_state, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[done]], device=device)

def act_env_eval(state, env_eval, current_net):
    with torch.no_grad():
        action = current_net(state).max(1)[1].view(1, 1)
    reward, done = env_eval.act(action)
    next_state = get_state(env_eval.state())
    return next_state, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[done]], device=device)


def train(sample, current_net, target_net, optimizer):

    batch_samples = transition(*zip(*sample))
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    dones = torch.cat(batch_samples.done)

    Q_s_a = current_net(states).gather(1, actions)
    none_done_next_state_index = torch.tensor([i for i, is_term in enumerate(dones) if is_term == 0], dtype=torch.int64, device=device)
    none_done_next_states = next_states.index_select(0, none_done_next_state_index)
    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)

    if len(none_done_next_states) != 0:
        policy_net_index = current_net(none_done_next_states).argmax(1).unsqueeze(1)
        Q_s_prime_a_prime[none_done_next_state_index] = torch.gather(target_net(none_done_next_states), 1, policy_net_index).detach()
    learning_target = rewards + GAMMA * Q_s_prime_a_prime

    loss = f.smooth_l1_loss(learning_target, Q_s_a)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return Q_s_a.mean().detach().cpu().numpy()



def Double_DQN(env, save_path, writer, eval_env):

    obs_dim = env.state_shape()[2]
    act_dim = env.num_actions()

    current_net = Q_ConvNet(obs_dim, act_dim).to(device)
    target_net = Q_ConvNet(obs_dim, act_dim).to(device)
    target_net.load_state_dict(current_net.state_dict())

    optimizer = optim.RMSprop(current_net.parameters(),
                              lr=LEARNING_RATE,
                              alpha=SQUARED_GRAD_MOMENTUM,
                              centered=True,
                              eps=MIN_SQUARED_GRAD)

    replay_buffer = tuple_replay_buffer(REPLAY_BUFFER_SIZE)

    avg_return = 0.0
    best_eval_score = 0.0

    step = 0
    episode = 0
    policy_net_update_counter = 0
    while step < NUM_FRAMES:
        score = 0.0
        env.reset()
        state = get_state(env.state())
        done = False
        while(not done) and step < NUM_FRAMES:
            next_state, action, reward, done = act_env(step, act_dim, state, env, current_net)
            replay_buffer.add(state, action, reward, next_state, done)

            sample = None
            if step > REPLAY_START_SIZE and len(replay_buffer.buffer) >= BATCH_SIZE:
                sample = replay_buffer.sample(BATCH_SIZE)
            if step % TRAINING_FREQ == 0 and sample is not None:
                policy_net_update_counter += 1
                Q_values = train(sample, current_net, target_net, optimizer)
            if policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(current_net.state_dict())

            score += reward.item()
            step += 1
            state = next_state
            if step % 10000 == 0:
                writer.add_scalar("AVR_RETURN/STEP", avg_return, global_step=step)
                writer.add_scalar("Q_VALUE/STEP", Q_values, global_step=step)
            if step % 50000 == 0:
                eval_scores = []
                for eval in range(EVAL_TIMES):
                    eval_score = 0
                    eval_env.reset()
                    state = get_state(env.state())
                    done = False
                    while (not done):
                        next_state, action, reward, done = act_env_eval(state, eval_env, current_net)
                        eval_score += reward.item()
                        state = next_state
                    eval_scores.append(eval_score)
                writer.add_scalar("EVALUATION/RETURN", np.mean(eval_scores), global_step=step)
                print("Evaluation: step={},score={}".format(step, np.mean(eval_scores)))
                torch.save(current_net.state_dict(), save_path+"{}_steps_checkpoint.pkl".format(step))
                if np.mean(eval_scores) > best_eval_score:
                    best_eval_score = np.mean(eval_scores)
                    torch.save(current_net.state_dict(), save_path + "best_checkpoint.pkl")
        episode += 1
        avg_return = 0.99 * avg_return + 0.01 * score

# fixed parameters
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 5000000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
LEARNING_RATE = 0.00025
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0
EVAL_TIMES = 10
GAMES = ['asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders']
RANDOM_SEEDS = range(5)

# specific device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    for game in GAMES:
        random_seeds = [0]
        for seed in random_seeds:
            save_path = os.getcwd() + '/RESULTS/Double_DQN/Env={},Seed={}/'.format(game, seed)
            writer = SummaryWriter(save_path)
            env = Environment(game)
            eval_env = Environment(game)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            Double_DQN(env, save_path, writer, eval_env)
            writer.close()