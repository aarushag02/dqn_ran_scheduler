import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Fixed-size circular experience replay buffer.

    Stores (state, action_idx, reward, next_state, done) tuples where
    action_idx is the integer index of the discrete template chosen by
    the DQN. Supports uniform random sampling for training.

    Parameters
    ----------
    capacity : int
        Maximum number of experiences. Oldest entry is overwritten when full.
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx: int, reward: float,
             next_state, done: bool):
        """
        Add one experience tuple to the buffer.

        Parameters
        ----------
        state      : array-like  shape (10,)
        action_idx : int         index into the DQN's template table
        reward     : float
        next_state : array-like  shape (10,)
        done       : bool
        """
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action_idx),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        """
        Return a random batch as stacked numpy arrays.

        Returns
        -------
        states      : np.ndarray  shape (B, 10)
        action_idxs : np.ndarray  shape (B,)  dtype int64
        rewards     : np.ndarray  shape (B,)
        next_states : np.ndarray  shape (B, 10)
        dones       : np.ndarray  shape (B,)  dtype float32
        """
        batch = random.sample(self.buffer, batch_size)
        states, action_idxs, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(action_idxs, dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.stack(next_states),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
