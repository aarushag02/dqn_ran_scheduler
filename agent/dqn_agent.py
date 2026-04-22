"""
dqn_agent.py — Discrete-action DQN with Double DQN and template-based action space.

The key fix over the original continuous-output version:
- DQN is a Q-learning algorithm designed for DISCRETE actions.
  Each output neuron must correspond to one specific, fixed action —
  not a component of a continuous allocation vector.
- We pre-generate N_TEMPLATES fixed PRB allocation vectors. The DQN
  outputs N_TEMPLATES Q-values and selects the index with the highest
  Q. The environment receives the corresponding allocation vector.
- The Bellman target is applied only to the Q-value of the action
  that was actually taken (via torch.gather), not broadcast to all
  outputs — this is the theoretically correct update.
- Double DQN: the online network selects the greedy next-action,
  the frozen target network evaluates it. This reduces Q overestimation.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Action template table
N_TEMPLATES = 64   # size of the discrete action space
N_UES       = 5
TOTAL_PRBS  = 50


def build_templates(n_templates: int = N_TEMPLATES,
                    n_ues: int = N_UES,
                    total_prbs: int = TOTAL_PRBS,
                    seed: int = 42) -> np.ndarray:
    """
    Generate a fixed table of PRB allocation vectors (templates).

    Each row is a valid allocation summing to total_prbs.
    The table is deterministic (seeded) so train and eval use identical
    templates without needing to save them alongside model weights.

    Structure
    1  equal split
    5  single-UE heavy   (one UE gets 70 %, rest share 30 %)
    5  single-UE moderate (one UE gets 50 %, rest share 50 %)
   10  two-UE heavy      (two UEs share 64 %, rest share 36 %)
   10  two-UE moderate   (two UEs share 48 %, rest share 52 %)
   33  random Dirichlet  (varied concentration for exploration diversity)

    Parameters
    n_templates : total number of templates (must be >= 31)
    n_ues       : number of UEs
    total_prbs  : PRB budget

    Returns
    np.ndarray  shape (n_templates, n_ues)  dtype float32
    """
    templates = []
    equal = total_prbs / n_ues

    # 1. equal split
    templates.append(np.full(n_ues, equal, dtype=np.float32))

    # 2. single-UE heavy: one UE gets 70 %, rest share 30
    heavy = 0.70 * total_prbs
    light = (total_prbs - heavy) / (n_ues - 1)
    for i in range(n_ues):
        t = np.full(n_ues, light, dtype=np.float32)
        t[i] = heavy
        templates.append(t)

    # 3. single-UE moderate: one UE gets 50 %, rest share 50
    mod = 0.50 * total_prbs
    rest_mod = (total_prbs - mod) / (n_ues - 1)
    for i in range(n_ues):
        t = np.full(n_ues, rest_mod, dtype=np.float32)
        t[i] = mod
        templates.append(t)

    # 4. two-UE heavy: pair shares 64 %, rest share 36
    pair_heavy = 0.32 * total_prbs   # each of the two UEs
    rest_heavy = (total_prbs - 2 * pair_heavy) / (n_ues - 2)
    for i in range(n_ues):
        for j in range(i + 1, n_ues):
            t = np.full(n_ues, rest_heavy, dtype=np.float32)
            t[i] = pair_heavy
            t[j] = pair_heavy
            templates.append(t)

    # 5. two-UE moderate: pair shares 48 %, rest share 52
    pair_mod = 0.24 * total_prbs
    rest_pmod = (total_prbs - 2 * pair_mod) / (n_ues - 2)
    for i in range(n_ues):
        for j in range(i + 1, n_ues):
            t = np.full(n_ues, rest_pmod, dtype=np.float32)
            t[i] = pair_mod
            t[j] = pair_mod
            templates.append(t)

    # 6. random Dirichlet fill
    rng = np.random.default_rng(seed)
    while len(templates) < n_templates:
        # mix of spiky (low alpha) and spread (high alpha) distributions
        alpha = rng.uniform(0.2, 3.0, n_ues)
        t = (rng.dirichlet(alpha) * total_prbs).astype(np.float32)
        templates.append(t)

    arr = np.stack(templates[:n_templates])   # (N, 5)
    # sanity: each row must sum to total_prbs
    assert np.allclose(arr.sum(axis=1), total_prbs, atol=1e-3), \
        "Template rows do not sum to total_prbs"
    return arr


# Q-network
class DQNNetwork(nn.Module):
    """
    Dueling Q-network: shared backbone → separate value V(s) and
    advantage A(s,a) streams.

        Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a))

    Separating value from advantage lets the network learn which states
    are inherently good independently of which action was taken.  This
    stabilises Q-value estimates and speeds up convergence, especially
    when many actions have similar Q-values (common early in training).

    Architecture
    ------------
    backbone  : state_dim → hidden → hidden   (ReLU)
    value     : hidden → hidden//2 → 1
    advantage : hidden → hidden//2 → n_actions
    """

    def __init__(self, state_dim: int = 10,
                 n_actions: int = N_TEMPLATES,
                 hidden_dim: int = 256):
        super().__init__()
        adv_hid = hidden_dim // 2
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, adv_hid),
            nn.ReLU(),
            nn.Linear(adv_hid, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, adv_hid),
            nn.ReLU(),
            nn.Linear(adv_hid, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)


# Agent
class DQNAgent:
    """
    Double DQN agent for discrete PRB template selection.

    Action space
    The agent maintains a table of N_TEMPLATES fixed PRB allocation
    vectors (see build_templates). At each step it selects an index
    and passes the corresponding vector to the environment.

    Learning
    Standard Double DQN (van Hasselt et al. 2016):
      - Online net selects the greedy next-action
      - Target net evaluates that action's Q-value
      - Loss is Huber on Q(s,a) vs Bellman target for the CHOSEN action only
        (implemented via torch.gather — not broadcast across all outputs)

    Parameters
    n_templates  : number of discrete allocation templates
    state_dim    : observation size (default 10)
    hidden_dim   : hidden layer width (default 256)
    lr           : Adam learning rate (default 5e-4)
    gamma        : discount factor (default 0.99)
    target_update: hard-copy online→target every N gradient steps
    device       : 'cpu', 'cuda', or 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        n_templates:   int   = N_TEMPLATES,
        state_dim:     int   = 10,
        hidden_dim:    int   = 256,
        lr:            float = 5e-4,
        gamma:         float = 0.99,
        target_update: int   = 500,
        n_step:        int   = 1,
        device:        str   = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device        = torch.device(device)
        self.gamma         = gamma
        self.gamma_n       = gamma ** n_step   # used in n-step Bellman target
        self.target_update = target_update
        self.n_templates   = n_templates
        self._learn_steps  = 0

        # pre-built, fixed action table (no gradients needed)
        self.templates = build_templates(n_templates)   # (N, 5) numpy

        # networks
        self.online = DQNNetwork(state_dim, n_templates, hidden_dim).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)


    # Public API
    def act(self, state: np.ndarray,
            epsilon: float = 0.0) -> tuple[int, np.ndarray]:
        """
        ε-greedy template selection.

        Parameters
        state   : np.ndarray  shape (10,)
        epsilon : float       exploration probability

        Returns
        action_idx : int         chosen template index
        prb_alloc  : np.ndarray  shape (5,)  the corresponding PRB vector
        """
        if np.random.random() < epsilon:
            idx = np.random.randint(self.n_templates)
        else:
            state_t = torch.tensor(state, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
            with torch.no_grad():
                idx = int(self.online(state_t).argmax(dim=1).item())

        return idx, self.templates[idx].copy()

    def learn(self, batch: tuple) -> float:
        """
        One Double-DQN gradient step.

        Parameters
        batch : (states, action_idxs, rewards, next_states, dones)
                action_idxs is np.ndarray of dtype int64

        Returns
        loss : float
        """
        states, action_idxs, rewards, next_states, dones = batch

        states_t      = torch.tensor(states,      dtype=torch.float32, device=self.device)
        action_idxs_t = torch.tensor(action_idxs, dtype=torch.int64,   device=self.device)
        rewards_t     = torch.tensor(rewards,      dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states,  dtype=torch.float32, device=self.device)
        dones_t       = torch.tensor(dones,        dtype=torch.float32, device=self.device)

        # Q-value of the action that was actually taken — shape (B,)
        q_all    = self.online(states_t)                                 # (B, N)
        q_chosen = q_all.gather(1, action_idxs_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Double DQN target:
        #   a* = argmax_a  Q_online(s', a)    ← online net selects action
        #   y  = r + γ * Q_target(s', a*)     ← target net evaluates it
        with torch.no_grad():
            next_actions = self.online(next_states_t).argmax(dim=1, keepdim=True)  # (B,1)
            next_q       = self.target(next_states_t).gather(1, next_actions).squeeze(1)  # (B,)
            targets      = rewards_t + self.gamma_n * next_q * (1.0 - dones_t)

        loss = F.huber_loss(q_chosen, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save(self.online.state_dict(), path)

    def load(self, path: str):
        self.online.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.target = copy.deepcopy(self.online)
        self.target.eval()
