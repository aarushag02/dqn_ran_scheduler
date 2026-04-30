import gymnasium as gym
import numpy as np
from gymnasium import spaces

"""
Custom Gymnasium environment simulating an O-RAN near-RT RIC control loop.

The environment models a single cell tower (gNB) serving N_UES mobile
users (UEs). At each time step the agent (DQN) receives the current
network state — CQI and PRB allocation per UE — and outputs a new PRB
allocation. The environment applies that allocation, computes per-UE
throughput using a simplified Shannon formula, and returns a scalar
reward equal to total throughput minus a fairness penalty.

Three traffic scenarios control how CQI values are initialised and
how they evolve over time:
    'uniform'       — all UEs similar, slow drift
    'heterogeneous' — UEs split across high/medium/low tiers
    'bursty'        — mostly uniform but with random 10% spike events
"""

class RANEnvironment(gym.Env):
    
    metadata = {"render_modes": []}

    def __init__(self, n_ues=5, total_prbs=50, scenario="uniform",
                 max_steps=200, min_throughput_floor=2.0,
                 min_prbs_per_ue=2):
        super().__init__()

        # core parameters
        self.n_ues = n_ues
        self.total_prbs = total_prbs
        self.scenario = scenario
        self.max_steps = max_steps
        self.min_prbs_per_ue = min_prbs_per_ue  # hard floor per UE

        # any UE falling below this throughput per step incurs a penalty
        self.min_throughput_floor = min_throughput_floor

        # gymnasium spaces
        # Observation: 3 × n_ues features, CQI-rank sorted.
        #   [0 : n_ues]       normalised CQI        in [0, 1]
        #   [n_ues : 2*n_ues] normalised PRB alloc  in [0, 1]
        #   [2*n_ues : 3*n_ues] relative throughput deficit  (unbounded, ~[-2, 2])
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(3 * n_ues, dtype=np.float32),
            high= np.inf * np.ones(3 * n_ues, dtype=np.float32),
            dtype=np.float32,
        )

        # Action: continuous PRB allocation per UE
        # Values are non-negative; environment normalises them to sum to total_prbs
        self.action_space = spaces.Box(low=0.0, high=float(total_prbs),
                                       shape=(n_ues,), dtype=np.float32)

        # internal state (initialised properly in reset())
        self.cqi       = np.zeros(n_ues, dtype=np.float32)
        self.prb_alloc = np.zeros(n_ues, dtype=np.float32)
        self.step_count = 0
        # permutation that maps CQI-rank order → physical UE order (set by _get_obs)
        self._sort_idx = np.arange(n_ues)
        # EMA throughput tracker for relative deficit computation (decay = 0.9)
        self._ema_tput = np.zeros(n_ues, dtype=np.float32)

    
    # PUBLIC API — called by the DQN training loop
    """
    Start a new episode. Initialise CQI values according to the
    chosen scenario and reset the step counter.
    Returns
    observation : np.ndarray  shape (10,)
    info        : dict        empty, required by gymnasium API
    """

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.cqi       = self._init_cqi()
        self.prb_alloc = np.full(self.n_ues,
                                 self.total_prbs / self.n_ues,
                                 dtype=np.float32)
        self.step_count = 0
        self._sort_idx = np.arange(self.n_ues)  # reset before first _get_obs call
        self._ema_tput = np.zeros(self.n_ues, dtype=np.float32)

        return self._get_obs(), {}

    """
    Apply the agent's PRB allocation, compute throughput and reward,
    then advance CQI by one Markov step.
    Parameters
        action : np.ndarray  shape (5,)  raw PRB allocation from DQN
    Returns
        observation : np.ndarray  shape (10,)
        reward      : float
        terminated  : bool   True when episode reaches max_steps
        truncated   : bool   always False (no time limit truncation)
        info        : dict   per-UE throughput for logging
    """
    def step(self, action):

        self.step_count += 1

        # 1. Unsort the action: the agent sees UEs sorted by CQI rank, so we
        #    must map its allocation back to physical UE indices before applying.
        unsort_idx = np.argsort(self._sort_idx)
        action = np.asarray(action, dtype=np.float32)[unsort_idx]

        # 2. apply hard per-UE minimum PRB floor then distribute the remainder
        # Each UE is guaranteed min_prbs_per_ue PRBs regardless of the action.
        # The action weights determine how the remaining budget is split.
        # This prevents the agent from starving any UE entirely and removes
        # the incentive to hide behind a trivial equal-split policy.
        action = np.clip(action, 0.0, None)
        action_sum = action.sum()
        if action_sum < 1e-6:
            action = np.ones(self.n_ues, dtype=np.float32)
            action_sum = float(self.n_ues)

        reserved  = self.min_prbs_per_ue * self.n_ues          # e.g. 10
        remaining = self.total_prbs - reserved                  # e.g. 40
        weights   = action / action_sum
        self.prb_alloc = (self.min_prbs_per_ue + weights * remaining).astype(np.float32)

        # 3. compute per-UE throughput
        throughputs = self._compute_throughput(self.prb_alloc, self.cqi)

        # 3a. update EMA throughput for relative deficit feature
        self._ema_tput = 0.9 * self._ema_tput + 0.1 * throughputs

        # 4. compute reward
        reward = self._compute_reward(throughputs)

        # 5. advance CQI (Markov step)
        self.cqi = self._update_cqi()

        # 6. check termination
        terminated = self.step_count >= self.max_steps
        truncated  = False

        info = {"throughputs": throughputs.tolist(),
                "cqi": self.cqi.tolist()}

        return self._get_obs(), float(reward), terminated, truncated, info

    # PRIVATE HELPERS
    def _get_obs(self):
        """
        Return a normalised, CQI-rank-sorted observation.

        UEs are sorted by descending CQI so that position 0 always refers to
        the UE with the best channel and position 4 to the worst.  This means
        the DQN only needs to learn one canonical allocation pattern
        ("give more PRBs to rank-0") rather than 5! physical-UE permutations.

        Both components are normalised to [0, 1]:
            CQI  : (raw - 1) / 14   maps [1, 15] → [0, 1]
            PRB  : raw / total_prbs maps [0, 50] → [0, 1]

        The sort permutation is stored in self._sort_idx so that step() can
        invert it when the agent's next action arrives.
        """
        self._sort_idx = np.argsort(self.cqi)[::-1].copy()   # descending CQI rank
        cqi_sorted = (self.cqi[self._sort_idx] - 1.0) / 14.0
        prb_sorted = self.prb_alloc[self._sort_idx] / self.total_prbs
        # relative deficit: (ema_i - mean_ema) / (mean_ema + ε)
        # positive = over-served, negative = under-served vs group average
        ema_sorted = self._ema_tput[self._sort_idx]
        mean_ema   = ema_sorted.mean()
        deficit    = (ema_sorted - mean_ema) / (mean_ema + 1e-6)
        return np.concatenate([cqi_sorted, prb_sorted, deficit]).astype(np.float32)

    #CQI initialisation
    def _init_cqi(self):
        """
        Initialise CQI values for all UEs based on the chosen scenario.
        CQI is always clamped to the integer range [1, 15].
        """
        if self.scenario == "uniform":
            # All UEs sampled from Normal(8, 2), clamped to [1, 15]
            cqi = self.np_random.normal(loc=8.0, scale=2.0,
                                        size=self.n_ues)

        elif self.scenario == "heterogeneous":
            #   UEs deliberately spread across quality tiers:
            #   UE 0, 1  → high   CQI ~ Uniform(10, 15)
            #   UE 2, 3  → medium CQI ~ Uniform(5, 9)
            #   UE 4     → low    CQI ~ Uniform(1, 4)
            cqi = np.zeros(self.n_ues)
            cqi[0:2] = self.np_random.uniform(10, 15, size=2)
            cqi[2:4] = self.np_random.uniform(5,  9,  size=2)
            cqi[4]   = self.np_random.uniform(1,  4)

        elif self.scenario == "bursty":
            # Start the same as uniform — bursts happen inside _update_cqi
            cqi = self.np_random.normal(loc=8.0, scale=2.0,
                                        size=self.n_ues)
        else:
            raise ValueError(
                f"Unknown scenario '{self.scenario}'. "
                "Choose 'uniform', 'heterogeneous', or 'bursty'."
            )

        return np.clip(np.round(cqi), 1, 15).astype(np.float32)

    # CQI evolution 
    """
    Advance CQI values by one time step using a scenario-specific
    Markov process. Returns updated CQI array clamped to [1, 15].
    """

    def _update_cqi(self):
        
        cqi = self.cqi.copy()

        if self.scenario == "uniform":
            # Small Gaussian drift — network is stable
            noise = self.np_random.normal(loc=0.0, scale=0.5,
                                          size=self.n_ues)
            cqi = cqi + noise

        elif self.scenario == "heterogeneous":
            # Drift exists but tiers are preserved — spread stays wide
            noise = self.np_random.normal(loc=0.0, scale=0.75,
                                          size=self.n_ues)
            cqi = cqi + noise

            # Soft pull back toward each UE's tier centre so tiers persist
            tier_centres = np.array([12.5, 12.5, 7.0, 7.0, 2.5],
                                     dtype=np.float32)
            cqi = cqi + 0.1 * (tier_centres - cqi)

        elif self.scenario == "bursty":
            # Normal drift for most steps
            noise = self.np_random.normal(loc=0.0, scale=0.5,
                                          size=self.n_ues)
            cqi = cqi + noise

            # 10% chance each UE experiences a sudden spike or drop
            burst_mask = self.np_random.random(self.n_ues) < 0.10
            burst_direction = self.np_random.choice([-1, 1],
                                                     size=self.n_ues)
            burst_magnitude = self.np_random.uniform(4, 8,
                                                      size=self.n_ues)
            cqi[burst_mask] += (burst_direction * burst_magnitude)[burst_mask]

        return np.clip(np.round(cqi), 1, 15).astype(np.float32)

    # Throughput 
    """
    Compute per-UE throughput using a simplified Shannon formula.
        throughput_i = prbs_i * log2(1 + SNR_i)

    CQI maps to SNR linearly:  SNR = (cqi - 1) * 2
        CQI 1  → SNR 0  → log2(1) = 0
        CQI 8  → SNR 14 → log2(15) ≈ 3.9
        CQI 15 → SNR 28 → log2(29) ≈ 4.9

    Parameters
        prbs : np.ndarray  shape (n_ues,)
        cqi  : np.ndarray  shape (n_ues,)

    Returns
        throughputs : np.ndarray  shape (n_ues,)
    """

    def _compute_throughput(self, prbs, cqi):
        
        snr = (cqi - 1) * 2.0
        throughputs = prbs * np.log2(1.0 + snr)
        return throughputs

    # Reward
    """
    Blended reward: α * PF_utility + (1-α) * linear_throughput

    α=0.9 keeps fairness as the dominant objective while the 10% linear
    term preserves a throughput gradient that discourages the agent from
    collapsing to the trivial equal-split policy.

    PF utility  = mean_i( log(1 + throughput_i) )  — proportional fairness
    Linear term = sum(throughputs) / total_prbs     — normalised aggregate
    """
    _ALPHA = 0.9   # PF weight; sweep confirmed this as the Pareto optimum

    def _compute_reward(self, throughputs):
        pf_utility = np.sum(np.log1p(throughputs)) / self.n_ues
        linear     = throughputs.sum() / self.total_prbs
        return self._ALPHA * pf_utility + (1.0 - self._ALPHA) * linear