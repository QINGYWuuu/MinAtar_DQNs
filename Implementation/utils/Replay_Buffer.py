import numpy as np
from collections import namedtuple
import random
from segement_tree import SumSegmentTree, MinSegmentTree
import torch

# normal replay buffer (s, a, r, s', d)
transition = namedtuple('transition', 'state, action, reward, next_state, done')
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

    def add(self, state, action, reward, next_state, done):
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

    def add(self, state, action, reward, next_state, done):
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
