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
                 max_steps=200, min_throughput_floor=2.0):
        super().__init__()

        # core parameters
        self.n_ues = n_ues
        self.total_prbs = total_prbs
        self.scenario = scenario
        self.max_steps = max_steps

        # any UE falling below this throughput per step incurs a penalty
        self.min_throughput_floor = min_throughput_floor

        # gymnasium spaces
        # State: [cqi_0..cqi_4, prb_0..prb_4]  →  10 floats
        # CQI range 1–15, PRB range 0–total_prbs
        low  = np.array([1.0]  * n_ues + [0.0]          * n_ues, dtype=np.float32)
        high = np.array([15.0] * n_ues + [total_prbs]   * n_ues, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float32)

        # Action: continuous PRB allocation per UE
        # Values are non-negative; environment normalises them to sum to total_prbs
        self.action_space = spaces.Box(low=0.0, high=float(total_prbs),
                                       shape=(n_ues,), dtype=np.float32)

        # internal state (initialised properly in reset())
        self.cqi      = np.zeros(n_ues, dtype=np.float32)
        self.prb_alloc = np.zeros(n_ues, dtype=np.float32)
        self.step_count = 0

    
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

        # 1. normalise action so PRBs sum exactly to total_prbs
        # The DQN output can be any positive numbers. We treat them as
        # weights and scale them. This guarantees the PRB budget is always
        # respected without requiring the DQN to learn that constraint.
        action = np.clip(action, 0.0, None)          # no negative PRBs
        action_sum = action.sum()
        if action_sum < 1e-6:
            # safety: if agent outputs all zeros, fall back to equal split
            action = np.ones(self.n_ues, dtype=np.float32)
            action_sum = float(self.n_ues)
        self.prb_alloc = (action / action_sum) * self.total_prbs

        # 2. compute per-UE throughput 
        throughputs = self._compute_throughput(self.prb_alloc, self.cqi)

        # 3. compute reward 
        reward = self._compute_reward(throughputs)

        # 4. advance CQI (Markov step)
        self.cqi = self._update_cqi()

        # 5. check termination 
        terminated = self.step_count >= self.max_steps
        truncated  = False

        info = {"throughputs": throughputs.tolist(),
                "cqi": self.cqi.tolist()}

        return self._get_obs(), float(reward), terminated, truncated, info

    # PRIVATE HELPERS
    def _get_obs(self):
        """
        Pack CQI and PRB allocation into a single flat state vector.
        Shape: (10,) — first 5 are CQI values, last 5 are PRB allocations.
        """
        return np.concatenate([self.cqi, self.prb_alloc]).astype(np.float32)

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
    Reward = total throughput across all UEs
            minus a penalty for each UE below the minimum floor.

    The fairness penalty discourages the DQN from maximising total
    throughput by starving weak UEs — a behaviour that would score
    well on throughput but produce a poor Jain's fairness index.

    Penalty per starved UE = 5.0 * shortfall below the floor.
    """

    def _compute_reward(self, throughputs):
        
        total_throughput = throughputs.sum()

        # identify UEs below the minimum throughput floor
        shortfalls = np.maximum(0.0, self.min_throughput_floor - throughputs)
        fairness_penalty = 5.0 * shortfalls.sum()

        reward = total_throughput - fairness_penalty
        return reward