import numpy as np
from collections import namedtuple
import random

# normal replay buffer (s, a, r, s', d)
transition = namedtuple('transition', 'state, action, reward, next_state, done')
class tuple_replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []
    def add(self, *args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)
        self.location = (self.location + 1) % self.buffer_size
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# mask replay buffer (s, a, r, s', d, m), e.g. bootstrapped
mask_transition = namedtuple('transition', 'state, action, reward, next_state, done, mask')
class mask_tuple_replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []
    def add(self, *args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(mask_transition(*args))
        else:
            self.buffer[self.location] = mask_transition(*args)
        self.location = (self.location + 1) % self.buffer_size
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
