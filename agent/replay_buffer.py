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
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action_idx),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
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


class NStepBuffer:
    """
    Accumulates n consecutive transitions and emits a single
    (s_0, a_0, G_n, s_n, done) tuple where G_n is the n-step
    discounted return:

        G_t^n = r_t + γ r_{t+1} + … + γ^{n-1} r_{t+n-1}

    The caller stores this in the main ReplayBuffer and uses γ^n
    (not γ) when computing the Bellman bootstrap target, since the
    reward already spans n environment steps.

    Episode boundaries are handled correctly: if the episode ends
    at step k < n the return is truncated and done=True is set so
    the bootstrap term is zeroed out in the loss.

    Usage
    -----
    At each step:
        t = nstep.push(s, a, r, s', done)
        if t: replay.push(*t)
    At episode end:
        for t in nstep.flush(): replay.push(*t)
        nstep.clear()          # safety — flush already drains the buf
    """

    def __init__(self, n: int, gamma: float):
        self.n     = n
        self.gamma = gamma
        self._buf  = deque()

    def push(self, state, action, reward, next_state, done):
        self._buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))
        if len(self._buf) < self.n:
            return None
        return self._make()

    def flush(self):
        out = []
        while self._buf:
            out.append(self._make())
        return out

    def clear(self):
        self._buf.clear()

    def _make(self):
        state, action = self._buf[0][0], self._buf[0][1]
        G = 0.0
        for i, (_, _, r, ns, d) in enumerate(self._buf):
            G += (self.gamma ** i) * r
            if d:
                self._buf.popleft()
                return (state, action, G, ns, True)
        _, _, _, next_state, done = self._buf[-1]
        self._buf.popleft()
        return (state, action, G, next_state, done)
