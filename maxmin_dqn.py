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

transition = namedtuple('transition', 'state, next_state, action, reward, done')

def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def act_env(step, replay_start_size, act_dim, state, env, dqns):

    if step < replay_start_size:
        action = torch.tensor([[random.randrange(act_dim)]], device=device)
    else:
        if step - replay_start_size >= FIRST_N_FRAMES:
            epsilon = END_EPSILON
        else:
            epsilon = ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (step - replay_start_size) + EPSILON
        if np.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(act_dim)]], device=device)
        else:
            minQ_pols = Qmin(dqns, Q_NUM, act_dim, state)
            with torch.no_grad():
                action = torch.argmax(minQ_pols).view(1, 1)
    reward, done = env.act(action)
    # Obtain s_prime
    next_state = get_state(env.state())
    return next_state, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[done]], device=device)

def Qmin(policy_solvers, Q_NUM, act_dim, state):
    finQ = torch.zeros(1, act_dim, device=device)
    for p in range(act_dim):
        selQ = torch.zeros(1, Q_NUM, device=device)
        for q in range(Q_NUM):
            selQ[0][q] = policy_solvers[q](state)[0][p]
        finQ[0][p] = torch.min(selQ)
    return finQ

def Qminbatch(target_solvers, Q_NUM, act_dim, states):
    stackedtens=torch.zeros(len(states), act_dim, Q_NUM, device=device)
    for i in range(Q_NUM):
        stackedtens[:, :, i] = target_solvers[i](states)
    finQ, fin_ind = stackedtens.min(2)
    return finQ

def train(sample, dqns, target_dqns, agent_id, act_dim, optimizers):
    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
    batch_samples = transition(*zip(*sample))

    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    dones = torch.cat(batch_samples.done)

    Q_s_a = dqns[agent_id](states).gather(1, actions)

    none_done_next_state_index = torch.tensor([i for i, is_term in enumerate(dones) if is_term == 0], dtype=torch.int64, device=device)
    none_done_next_states = next_states.index_select(0, none_done_next_state_index)

    mintargpols = Qminbatch(target_dqns, Q_NUM, act_dim, next_states)
    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
    if len(none_done_next_states) != 0:
        Q_s_prime_a_prime[none_done_next_state_index] = mintargpols[none_done_next_state_index].detach().max(1)[0].unsqueeze(1)
    target = rewards + GAMMA * Q_s_prime_a_prime
    # Huber loss
    loss = f.smooth_l1_loss(target, Q_s_a)
    optimizers[agent_id].zero_grad()
    loss.backward()
    optimizers[agent_id].step()


def maxmin_dqn_agent(env, save_path, writer):
    # Get channels and number of actions specific to each game
    obs_dim = env.state_shape()[2]
    act_dim = env.num_actions()

    # Instantiate networks, optimizer, loss and buffer
    dqns = [Q_ConvNet(obs_dim, act_dim).to(device) for k in range(Q_NUM)]
    target_dqns = [Q_ConvNet(obs_dim, act_dim).to(device) for k in range(Q_NUM)]

    optimizers = []
    for i in range(Q_NUM):
        target_dqns[i].load_state_dict(dqns[i].state_dict())
        optimizers.append(optim.RMSprop(dqns[i].parameters(), lr=LEARNING_RATE, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD))

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
    # reward_seq = []

    while step < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        score = 0.0
        # Initialize the environment and start state
        env.reset()
        state = get_state(env.state())
        done = False
        while (not done) and step < NUM_FRAMES:
            next_state, action, reward, done = act_env(step, replay_start_size, act_dim, state, env, dqns)
            replay_buffer.add(state, action, reward, next_state, done)

            sample = None
            if step > REPLAY_START_SIZE and len(replay_buffer.buffer) >= BATCH_SIZE:
                sample = replay_buffer.sample(BATCH_SIZE)
            agent_id = np.random.randint(Q_NUM)
            # Train every n number of frames defined by TRAINING_FREQ
            if step % TRAINING_FREQ == 0 and sample is not None:
                policy_net_update_counter += 1
                train(sample, dqns, target_dqns, agent_id, act_dim, optimizers)

            # Update the target network only after some number of policy network updates
            if policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_dqns[agent_id].load_state_dict(dqns[agent_id].state_dict())

            score += reward.item()
            step += 1
            state = next_state
            if step % 10000 == 0:
                writer.add_scalar("AVR_RETURN/STEP", avg_return, global_step=step)

        episode += 1
        data_return.append(score)
        frame_stamp.append(step)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * score
        if episode % 1000 == 0:
            logging.info("Episode " + str(episode) + " | Return: " + str(score) + " | Avg return: " +
                         str(np.around(avg_return, 2)) + " | Frame: " + str(step) + " | Time per frame: " + str((time.time() - t_start) / step))

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

Q_NUM = 5

device = torch.device("cuda")


if __name__ == '__main__':
    # breakout, space_invaders, asterix, freeway, seaquest
    for game in ['breakout']:
        Nruns = 1
        for run_no in range(Nruns):
            logging.basicConfig(level=logging.INFO)
            save_path = os.getcwd() + "/" + game + '/maxmin_dqn'
            writer = SummaryWriter(save_path+'/{}_log'.format(run_no))
            env = Environment(game)
            maxmin_dqn_agent(env, save_path, writer)
