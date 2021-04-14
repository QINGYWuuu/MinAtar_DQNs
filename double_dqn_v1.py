import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from minatar import Environment

import random, logging, os, time
import numpy as np
from collections import namedtuple

from utils.Replay_Buffer import tuple_replay_buffer
from utils.Agent_Nets import Q_ConvNet
from tensorboardX import SummaryWriter

transition = namedtuple('transition', 'state, action, reward, next_state, done')

def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()

def act_env(step, replay_start_size, act_dim, state, env, policy_net):
    if step < replay_start_size:
        action = torch.tensor([[random.randrange(act_dim)]], device=device)
    else:
        if step - replay_start_size >= FIRST_N_FRAMES:
            epsilon = END_EPSILON
        else:
            epsilon = ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (step - replay_start_size) + EPSILON

        # epsilon greedy
        if np.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(act_dim)]], device=device)
        else:
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
    reward, done = env.act(action)
    next_state = get_state(env.state())
    return next_state, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[done]], device=device)



def train(sample, policy_net, target_net, optimizer):
    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
    batch_samples = transition(*zip(*sample))

    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    dones = torch.cat(batch_samples.done)

    Q_s_a = policy_net(states).gather(1, actions)

    none_done_next_state_index = torch.tensor([i for i, is_term in enumerate(dones) if is_term == 0], dtype=torch.int64, device=device)
    none_done_next_states = next_states.index_select(0, none_done_next_state_index)
    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)

    if len(none_done_next_states) != 0:
        ###double dqn###
        policy_net_index = policy_net(none_done_next_states).argmax(1).unsqueeze(1)
        Q_s_prime_a_prime[none_done_next_state_index] = torch.gather(target_net(none_done_next_states), 1, policy_net_index).detach()
    target = rewards + GAMMA * Q_s_prime_a_prime
    # Huber loss
    loss = f.smooth_l1_loss(target, Q_s_a)
    # Zero gradients, backprop, update the weights of policy_net
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def double_dqn_agent_v1(env, save_path, writer):

    obs_dim = env.state_shape()[2]
    act_dim = env.num_actions()

    # Instantiate networks, optimizer
    dqn = Q_ConvNet(obs_dim, act_dim).to(device)
    dqn_target = Q_ConvNet(obs_dim, act_dim).to(device)
    dqn_target.load_state_dict(dqn.state_dict())

    optimizer = optim.RMSprop(dqn.parameters(), lr=LEARNING_RATE, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)

    # Replay Buffer
    replay_buffer = tuple_replay_buffer(REPLAY_BUFFER_SIZE)
    replay_start_size = REPLAY_START_SIZE

    # Data containers for performance measure and model related data
    data_return = []
    frame_stamp = []
    avg_return = 0.0

    # Train for a number of frames
    step = 0
    episode = 0
    policy_net_update_counter = 0
    t_start = time.time()
    reward_seq = []
    while step < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        score = 0.0

        # Initialize the environment and start state
        env.reset()
        state = get_state(env.state())
        done = False
        while(not done) and step < NUM_FRAMES:
            next_state, action, reward, done = act_env(step, replay_start_size, act_dim, state, env, dqn)
            replay_buffer.add(state, action, reward, next_state, done)

            sample = None
            if step > REPLAY_START_SIZE and len(replay_buffer.buffer) >= BATCH_SIZE:
                sample = replay_buffer.sample(BATCH_SIZE)

            if step % TRAINING_FREQ == 0 and sample is not None:
                policy_net_update_counter += 1
                train(sample, dqn, dqn_target, optimizer)

            # Update the target network
            if policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                dqn_target.load_state_dict(dqn.state_dict())

            score += reward.item()
            reward_seq.append(reward.item())
            step += 1
            state = next_state
            if step % 10000 == 0:
                writer.add_scalar("AVR_RETURN/STEP", avg_return, global_step=step)
            # if episode % 100 == 0:
            #     writer.add_scalar("AVR_RETURN/EPISODE", avg_return, global_step=episode)

        episode += 1
        data_return.append(score)
        frame_stamp.append(step)

        avg_return = 0.99 * avg_return + 0.01 * score
        if episode % 1000 == 0:
            logging.info("Episode " + str(episode) + " | Return: " + str(score) + " | Avg return: " +
                         str(np.around(avg_return, 2)) + " | Frame: " + str(step)+" | Time per frame: " +str((time.time()-t_start)/step) )

    # Print final logging info
    logging.info("Avg return: " + str(np.around(avg_return, 2)) + " | Time per frame: " + str((time.time()-t_start)/step))

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
device = torch.device("cuda")

if __name__ == '__main__':
    # breakout, space_invaders, asterix, freeway, seaquest
    for game in ['breakout']:
        random_seeds = [4]
        for seed in random_seeds:

            logging.basicConfig(level=logging.INFO)
            save_path = os.getcwd() + "/" + game + '/double_dqn_v1'
            writer = SummaryWriter(save_path+'/seed={}'.format(seed))
            env = Environment(game)

            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            double_dqn_agent_v1(env, save_path, writer)