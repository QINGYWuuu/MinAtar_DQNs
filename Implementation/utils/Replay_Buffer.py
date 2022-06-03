import numpy as np
from collections import namedtuple
import random
from utils.segement_tree import SumSegmentTree, MinSegmentTree
import torch

from minatar import Environment



# normal replay buffer (s, a, r, s', d)
class replay_buffer:
    def __init__(self, buffer_size, state_dim, args):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer_len = 0
        self.states = torch.zeros((buffer_size,) + tuple(state_dim), device=args.device)
        self.actions = torch.zeros((buffer_size,) + (1, 1), device=args.device).long()
        self.rewards = torch.zeros((buffer_size,) + (1, 1), device=args.device)
        self.next_states = torch.zeros((buffer_size,) + tuple(state_dim), device=args.device)
        self.dones = torch.zeros((buffer_size,) + (1, 1), device=args.device).bool()

    def store(self, state, action, reward, next_state, done):
        assert self.buffer_len <= self.buffer_size
        self.states[self.location] = state
        self.actions[self.location] = action
        self.rewards[self.location] = reward
        self.next_states[self.location] = next_state
        self.dones[self.location] = done
        if self.buffer_len < self.buffer_size:
           self.buffer_len += 1
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        assert self.buffer_len <= self.buffer_size
        assert self.buffer_len >= batch_size
        indices = torch.LongTensor(np.random.randint(0, self.buffer_len, size=batch_size))
        return (self.states[indices], \
               self.actions[indices], \
               self.rewards[indices], \
               self.next_states[indices], \
               self.dones[indices])


class multi_step_replay_buffer:
    def __init__(self, buffer_size, state_dim, args):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer_len = 0
        self.states = torch.zeros((buffer_size,) + tuple(state_dim), device=args.device)
        self.actions = torch.zeros((buffer_size,) + (1, 1), device=args.device).long()
        self.rewards = torch.zeros((buffer_size, args.n_step,) + (1, 1), device=args.device)
        self.next_states = torch.zeros((buffer_size, args.n_step,) + tuple(state_dim), device=args.device)
        self.dones = torch.zeros((buffer_size,) + (1, 1), device=args.device).bool()
        self.max_steps = torch.zeros((buffer_size,) + (1, 1), device=args.device).long() # to compute the transition last n step

    def store(self, state, action, reward, next_state, done, max_step):
        assert self.buffer_len <= self.buffer_size
        self.states[self.location] = state
        self.actions[self.location] = action
        self.rewards[self.location] = reward
        self.next_states[self.location] = next_state
        self.dones[self.location] = done
        self.max_steps[self.location] = max_step
        if self.buffer_len < self.buffer_size:
           self.buffer_len += 1
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        assert self.buffer_len <= self.buffer_size
        assert self.buffer_len >= batch_size
        indices = torch.LongTensor(np.random.randint(0, self.buffer_len, size=batch_size))
        return (self.states[indices], \
               self.actions[indices], \
               self.rewards[indices], \
               self.next_states[indices], \
               self.dones[indices], \
               self.max_steps[indices])

class prioritized_replay_buffer:
    def __init__(self, buffer_size, state_dim, args, alpha=0.6, batch_size=32):
        assert alpha >= 0
        self.buffer_size = buffer_size
        self.location = 0
        self.batch_size = batch_size
        self.buffer_len = 0
        self.states = torch.zeros((buffer_size,) + tuple(state_dim), device=args.device)
        self.actions = torch.zeros((buffer_size,) + (1, 1), device=args.device).long()
        self.rewards = torch.zeros((buffer_size,) + (1, 1), device=args.device)
        self.next_states = torch.zeros((buffer_size,) + tuple(state_dim), device=args.device)
        self.dones = torch.zeros((buffer_size,) + (1, 1), device=args.device).bool()

        self.alpha = alpha
        self.max_priority, self.tree_ptr = 1.0, 0
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, state, action, reward, next_state, done):
        assert self.buffer_len <= self.buffer_size
        self.states[self.location] = state
        self.actions[self.location] = action
        self.rewards[self.location] = reward
        self.next_states[self.location] = next_state
        self.dones[self.location] = done
        if self.buffer_len < self.buffer_size:
            self.buffer_len += 1
        self.location = (self.location + 1) % self.buffer_size

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

    def sample(self, batch_size, beta=0.4):
        assert self.buffer_len <= self.buffer_size
        assert self.buffer_len >= batch_size
        assert beta >= 0

        indices = self._sample_proportional()
        samples = (self.states[indices], \
               self.actions[indices], \
               self.rewards[indices], \
               self.next_states[indices], \
               self.dones[indices])
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, self.buffer_len - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.buffer_len) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.buffer_len) ** (-beta)
        weight = weight / max_weight

        return weight



# traj replay buffer (s, a, r, d)
class traj_replay_buffer:
    def __init__(self, buffer_size, state_dim, gamma, clip_step):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.buffer_len = 0
        self.full = False
        self.states = torch.zeros((buffer_size,) + tuple(state_dim))
        self.actions = torch.zeros((buffer_size,) + (1, 1)).long()
        self.rewards = torch.zeros((buffer_size,) + (1, 1))
        self.dones = torch.zeros((buffer_size,) + (1, 1)).bool()
        self.ends = torch.zeros((buffer_size,) + (1, 1)).long() # the final ptr of transition

        self.traj_max_len = 100
        self.gamma = gamma
        self.Generate_Gamma_Vector(self.traj_max_len)

        self.clip_step = clip_step

    def Generate_Gamma_Vector(self, traj_max_len):
        # gamma_vec
        # [1, gamma, gamma2,... , gammaN]
        self.gamma_vec = torch.tensor((np.ones(traj_max_len))).float()
        for i in range(traj_max_len):
            self.gamma_vec[i] = self.gamma**i

        self.traj_max_len = traj_max_len

    def store(self, traj_states, traj_actions, traj_rewards, traj_dones):
        len_traj = len(traj_states)
        if self.ptr + len_traj < self.buffer_size:
            self.states[self.ptr: self.ptr + len_traj] = traj_states
            self.actions[self.ptr: self.ptr + len_traj - 1] = traj_actions
            self.rewards[self.ptr: self.ptr + len_traj - 1] = traj_rewards
            self.dones[self.ptr: self.ptr + len_traj] = traj_dones
            self.ends[self.ptr: self.ptr + len_traj] = self.ptr + len_traj

            self.ptr += len_traj

            if self.full == False:
                self.buffer_len = self.ptr

        elif self.ptr + len_traj == self.buffer_size:
            temp_len1 = self.buffer_size - self.ptr
            self.states[self.ptr:] = traj_states
            self.actions[self.ptr:-1] = traj_actions
            self.rewards[self.ptr:-1] = traj_rewards
            self.dones[self.ptr:] = traj_dones
            self.ends[self.ptr:] = 0
            self.ptr = 0
            if self.full == False:
                self.full = True
                self.buffer_len = self.buffer_size

        else:
            temp_len1 = self.buffer_size - self.ptr
            self.states[self.ptr:] = traj_states[: temp_len1]
            self.actions[self.ptr:] = traj_actions[: temp_len1]
            self.rewards[self.ptr:] = traj_rewards[: temp_len1]
            self.dones[self.ptr:] = traj_dones[: temp_len1]

            temp_len2 = len_traj - (self.buffer_size - self.ptr)
            self.states[: temp_len2] = traj_states[temp_len1: ]
            self.actions[: temp_len2-1] = traj_actions[temp_len1: ]
            self.rewards[: temp_len2-1] = traj_rewards[temp_len1: ]
            self.dones[: temp_len2] = traj_dones[temp_len1: ]

            self.ends[self.ptr:] = temp_len2
            self.ends[: temp_len2] = temp_len2

            self.ptr = temp_len2
            if self.full == False:
                self.full = True
                self.buffer_len = self.buffer_size

    def sample(self, batch_size, Target_Net):
        sampled_targets = []

        while True: # resample while having a wrong sample
            indices = torch.LongTensor(np.random.randint(0, self.buffer_len, size=batch_size))
            if torch.sum((self.ends[indices].squeeze() - indices) <= 1) == 0:
                break

        sampled_states = self.states[indices]
        sampled_actions = self.actions[indices]

        # computing the target value directly
        for idx in indices:
            end_idx = self.ends[idx]
            if end_idx > idx:
                n_step_rewards = self.rewards[idx: end_idx-1].squeeze()
                n_step_next_states = self.states[idx+1: end_idx]
            else:
                n_step_rewards = torch.cat((self.rewards[idx:], self.rewards[: end_idx-1])).squeeze()
                n_step_next_states = torch.cat((self.states[idx+1:], self.states[: end_idx]))

            traj_len = n_step_next_states.size()[0]
            if traj_len > self.traj_max_len:
                self.Generate_Gamma_Vector(traj_len+100)

            with torch.no_grad():
                n_step_target = Target_Net(n_step_next_states.squeeze(dim=1)).detach().max(1)[0]
                n_step_target[-1] = 0

            # discount return  [r0, gamma r1, gamma2 r2, ...]
            discount_n_step_return = self.gamma_vec[:traj_len] * n_step_rewards
            # discount return  [gamma Qs1, gamma2 Qs2, ...]
            discount_n_step_target = self.gamma * self.gamma_vec[:traj_len] * n_step_target
            # discount return  [r0, r0+gamma r1, r0+gamma r1+gamma2 r2, ...]
            for r_i in range(1, traj_len):
                discount_n_step_return[r_i] += discount_n_step_return[r_i-1]
            target_values = discount_n_step_return + discount_n_step_target

            if self.clip_step == None:
                n_step = target_values.argmax().numpy().tolist()
            else:
                n_step = min(self.clip_step, target_values.argmax().numpy().tolist())
            sampled_targets.append(target_values[n_step].numpy())

        sampled_targets = torch.tensor(np.array(sampled_targets))
        return (sampled_states, sampled_actions, sampled_targets)


class prioritized_traj_replay_buffer:
    def __init__(self, buffer_size, state_dim, gamma, clip_step, alpha=0.6):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.buffer_len = 0
        self.full = False
        self.states = torch.zeros((buffer_size,) + tuple(state_dim))
        self.actions = torch.zeros((buffer_size,) + (1, 1)).long()
        self.rewards = torch.zeros((buffer_size,) + (1, 1))
        self.dones = torch.zeros((buffer_size,) + (1, 1)).bool()
        self.ends = torch.zeros((buffer_size,) + (1, 1)).long() # the final ptr of transition

        self.traj_max_len = 100
        self.gamma = gamma
        self.Generate_Gamma_Vector(self.traj_max_len)

        self.clip_step = clip_step

        self.alpha = alpha
        self.max_priority, self.tree_ptr = 1.0, 0
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def Generate_Gamma_Vector(self, traj_max_len):
        # gamma_vec
        # [1, gamma, gamma2,... , gammaN]
        self.gamma_vec = torch.tensor((np.ones(traj_max_len))).float()
        for i in range(traj_max_len):
            self.gamma_vec[i] = self.gamma**i

        self.traj_max_len = traj_max_len

    def store(self, traj_states, traj_actions, traj_rewards, traj_dones):
        len_traj = len(traj_states)
        if self.ptr + len_traj < self.buffer_size:
            self.states[self.ptr: self.ptr + len_traj] = traj_states
            self.actions[self.ptr: self.ptr + len_traj - 1] = traj_actions
            self.rewards[self.ptr: self.ptr + len_traj - 1] = traj_rewards
            self.dones[self.ptr: self.ptr + len_traj] = traj_dones
            self.ends[self.ptr: self.ptr + len_traj] = self.ptr + len_traj

            self.ptr += len_traj

            self.sum_tree[self.tree_ptr: self.tree_ptr + len_traj] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr: self.tree_ptr + len_traj] = self.max_priority ** self.alpha
            self.tree_ptr += len_traj

            if self.full == False:
                self.buffer_len = self.ptr
        else:
            temp_len1 = self.buffer_size - self.ptr
            self.states[self.ptr:] = traj_states[: temp_len1]
            self.actions[self.ptr:] = traj_actions[: temp_len1]
            self.rewards[self.ptr:] = traj_rewards[: temp_len1]
            self.dones[self.ptr:] = traj_dones[: temp_len1]

            self.sum_tree[self.tree_ptr:] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr:] = self.max_priority ** self.alpha


            temp_len2 = len_traj - (self.buffer_size - self.ptr)
            self.states[: temp_len2] = traj_states[temp_len1: ]
            self.actions[: temp_len2-1] = traj_actions[temp_len1: ]
            self.rewards[: temp_len2-1] = traj_rewards[temp_len1: ]
            self.dones[: temp_len2] = traj_dones[temp_len1: ]

            self.sum_tree[: temp_len2] = self.max_priority ** self.alpha
            self.min_tree[: temp_len2] = self.max_priority ** self.alpha

            self.ends[self.ptr:] = temp_len2
            self.ends[: temp_len2] = temp_len2

            self.ptr = temp_len2
            self.tree_ptr = temp_len2

            if self.full == False:
                self.full = True
                self.buffer_len = self.buffer_size

    def sample(self, batch_size, Target_Net, beta=0.4):
        sampled_targets = []

        while True: #  having a wrong sample
            indices = self._sample_proportional()
            if torch.sum((self.ends[indices].squeeze() - indices) <= 1) == 0:
                break
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        sampled_states = self.states[indices]
        sampled_actions = self.actions[indices]

        # computing the target value directly
        for idx in indices:
            end_idx = self.ends[idx]
            if end_idx > idx:
                n_step_rewards = self.rewards[idx: end_idx-1].squeeze()
                n_step_next_states = self.states[idx+1: end_idx]
            else:
                n_step_rewards = torch.cat((self.rewards[idx:], self.rewards[: end_idx-1])).squeeze()
                n_step_next_states = torch.cat((self.states[idx+1:], self.states[: end_idx]))

            traj_len = n_step_next_states.size()[0]
            if traj_len > self.traj_max_len:
                self.Generate_Gamma_Vector(traj_len+100)

            with torch.no_grad():
                n_step_target = Target_Net(n_step_next_states.squeeze(dim=1)).detach().max(1)[0]
                n_step_target[-1] = 0

            # discount return  [r0, gamma r1, gamma2 r2, ...]
            discount_n_step_return = self.gamma_vec[:traj_len] * n_step_rewards
            # discount return  [gamma Qs1, gamma2 Qs2, ...]
            discount_n_step_target = self.gamma * self.gamma_vec[:traj_len] * n_step_target
            # discount return  [r0, r0+gamma r1, r0+gamma r1+gamma2 r2, ...]
            for r_i in range(1, traj_len):
                discount_n_step_return[r_i] += discount_n_step_return[r_i-1]
            target_values = discount_n_step_return + discount_n_step_target

            if self.clip_step == None:
                sampled_targets.append(target_values.max().numpy())
            else:
                n_step = min(self.clip_step, target_values.argmax().numpy().tolist())
                sampled_targets.append(target_values[n_step].numpy())

        sampled_targets = torch.tensor(np.array(sampled_targets))
        return (sampled_states, sampled_actions, sampled_targets), weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, self.buffer_len - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.buffer_len) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.buffer_len) ** (-beta)
        weight = weight / max_weight

        return weight