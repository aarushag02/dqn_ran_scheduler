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


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer — Schaul et al. 2016.

    Experiences are sampled proportional to their TD-error priority so
    that rare, high-surprise transitions (e.g. successfully helping a
    starved UE) are replayed more often than routine near-equal steps.

    Priority for a new experience is initialised to max(existing) so it
    is guaranteed to be sampled at least once before being deprioritised.

    Importance-sampling (IS) weights correct for the introduced sampling
    bias; β is annealed from β_start → 1.0 over the course of training.

    Parameters
    ----------
    capacity  : int    maximum number of stored experiences
    alpha     : float  priority exponent  (0 = uniform, 1 = full priority)
    beta_start: float  initial IS-weight exponent
    beta_steps: int    steps over which β is annealed to 1.0
    """

    def __init__(self, capacity: int = 50_000,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_steps: int = 100_000):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self._step      = 0

        self.buffer     = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self._pos       = 0   # circular write pointer

    # ── Sum-tree helpers ──────────────────────────────────────────────────────

    def _max_priority(self) -> float:
        if len(self.buffer) == 0:
            return 1.0
        return float(self.priorities[:len(self.buffer)].max())

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, state, action_idx: int, reward: float,
             next_state, done: bool):
        # capture max priority before mutating buffer to avoid reading own zero
        max_p = self._max_priority()
        experience = (
            np.array(state,      dtype=np.float32),
            int(action_idx),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self._pos] = experience

        # new transitions get max priority so they're sampled at least once
        self.priorities[self._pos] = max_p
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=probs)

        # IS weights: w_i = (1 / N·P(i))^β  normalised by max weight
        beta = min(1.0, self.beta_start +
                   self._step * (1.0 - self.beta_start) / self.beta_steps)
        self._step += 1

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, action_idxs, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(action_idxs, dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.stack(next_states),
            np.array(dones,       dtype=np.float32),
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Call after each learning step with |TD error| for sampled transitions."""
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + 1e-6   # small ε prevents zero

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
