# DQN-Based Dynamic PRB Allocation for O-RAN Near-RT RIC

**Columbia University — ML in Networks**<br>
Aarush Agarwal · Arka Khorchidian · Keerthilakshmi Sivakumar

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to dynamically allocate Physical Resource Blocks (PRBs) inside an O-RAN near Real-Time RAN Intelligent Controller (near-RT RIC) environment. The agent is trained entirely in simulation using a custom OpenAI Gymnasium environment and evaluated against two rule-based baselines — Round-Robin and Proportional-Fair — across three traffic scenarios.

The DQN achieves the highest aggregate throughput in all three scenarios, with gains of up to **26% over Round-Robin** and **4.5% over Proportional-Fair**, demonstrating that a learned policy can outperform fixed-formula schedulers without any prior knowledge of the channel model.

---

## Key Results

| Scenario | Round-Robin | Prop-Fair | DQN |
|---|---|---|---|
| Uniform | 134.8 | 162.8 | **170.1** |
| Heterogeneous | 175.8 | 199.2 | **206.5** |
| Bursty | 163.3 | 199.6 | **201.5** |

*Metric: average total throughput (arbitrary units) over 500 evaluation steps.*

---

## Project Structure

```
dqn_ran_scheduler/
│
├── env/
│   ├── __init__.py
│   └── ran_environment.py
│
├── agent/
│   ├── __init__.py
│   ├── dqn_agent.py
│   └── replay_buffer.py
│
├── eval/
│   ├── __init__.py
│   ├── baselines.py
│   └── metrics.py
│
├── models/
│   └── dqn_trained_{scenario}.pt
│
├── notebooks/
│   ├── results.ipynb
│   └── comparison.ipynb
│
├── train.py
├── train_sb3.py
├── demo.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Environment Design

The environment (`env/ran_environment.py`) simulates a single-cell gNodeB serving **5 UEs** with a total PRB budget of **50 PRBs** per time step.

### State Space (15-dimensional)
Each observation concatenates three blocks, sorted in descending CQI rank order:
- **Normalised CQI**: `(q_i − 1) / 14 ∈ [0, 1]`
- **Normalised PRB allocation**: `p_i / P ∈ [0, 1]`
- **Relative EMA throughput deficit**: how far each UE's recent throughput is from the network average

Sorting by CQI rank reduces the effective state space from 120 physical permutations to a single canonical ordering.

### Action Space (64 discrete templates)

The agent selects from 64 pre-built PRB allocation templates rather than outputting continuous values, preserving the theoretical correctness of Q-learning.

| Template Type | Count | Description |
|---|---|---|
| Equal split | 1 | 10 PRBs per UE |
| Single-UE heavy | 5 | 40% to one UE |
| Single-UE moderate | 5 | 30% to one UE |
| Two-UE heavy | 10 | 32% each to a pair |
| Two-UE moderate | 10 | 24% each to a pair |
| Dirichlet random | 33 | α ~ U(0.2, 3.0), seeded |

A hard minimum floor of **2 PRBs per UE** is enforced architecturally — starvation is structurally impossible.

### Reward Function
r = 0.9 × (1/N) × Σ ln(1 + Tᵢ)  +  0.1 × ΣTᵢ / P

The dominant proportional-fairness term (0.9) encodes the fairness objective. The linear term (0.1) preserves a throughput gradient that prevents collapse to trivial equal-split policies.

### Traffic Scenarios

| Scenario | CQI Initialisation | CQI Evolution |
|---|---|---|
| Uniform | N(8, 2) for all UEs | Slow Gaussian drift (σ=0.5) |
| Heterogeneous | High / medium / low tiers | Drift + soft mean-reversion to tier |
| Bursty | N(8, 2) for all UEs | Slow drift + 10% spike probability (±4–8 units) |

---

## DQN Agent Design

The agent (`agent/dqn_agent.py`) uses a **Dueling Double DQN** with **n-step returns**.

### Architecture
Input (15) → FC(256) → ReLU → FC(256) → ReLU
↓
Value stream:     FC(128) → scalar V(s)
Advantage stream: FC(128) → 64 values A(s,a)
↓
Q(s,a) = V(s) + A(s,a) − mean(A(s,·))

### Key Design Decisions

- **Double DQN** — online network selects actions, frozen target network evaluates them, reducing Q-value overestimation
- **n-step returns (n=3)** — accumulates 3 steps of reward before bootstrapping, improving credit assignment
- **Huber loss** — applied only to the Q-value of the action actually taken
- **MPS support** — automatically uses Apple Silicon GPU acceleration on M-series MacBooks

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Total training steps | 100,000 |
| Batch size | 64 |
| Learning rate (Adam) | 1×10⁻⁴ |
| Discount factor γ | 0.95 |
| n-step return | 3 |
| Replay buffer capacity | 50,000 |
| Warmup steps | 1,000 |
| ε initial / minimum | 1.0 / 0.05 |
| ε decay | 0.9997 |
| Target network update | every 200 gradient steps |
| Gradient clipping | max norm 1.0 |

---

## Installation

**Requirements:** Python 3.11+, macOS / Linux (Apple Silicon natively supported)

```bash
# clone the repository
git clone https://github.com/aarushag02/dqn_ran_scheduler.git
cd dqn_ran_scheduler

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# verify installation
python3 -c "import gymnasium, torch, numpy, matplotlib; print('All good')"
```

---

## Usage

### Train the DQN

```bash
# default scenario (uniform)
python train.py

# specific scenario
python train.py --scenario heterogeneous
python train.py --scenario bursty

# full options
python train.py --scenario bursty --total-steps 100000 --lr 1e-4
```

Trained weights are saved to `models/dqn_trained_{scenario}.pt`.

### Train PPO or DDPG (optional)

```bash
pip install stable-baselines3

python train_sb3.py --algo ppo  --scenario uniform
python train_sb3.py --algo ddpg --scenario heterogeneous
python train_sb3.py --algo ppo  --scenario bursty --total-steps 100000
```

### Run the live demo

```bash
python demo.py
```

Opens a real-time animation showing PRB allocations per UE, live CQI values, and cumulative throughput for DQN vs Round-Robin side by side.

### Generate result plots

```bash
jupyter notebook notebooks/results.ipynb      # four core results
jupyter notebook notebooks/comparison.ipynb   # five-algorithm comparison
```

---

## Results

### Summary Table

| Scenario | Scheduler | Avg. Throughput | Jain's Index |
|---|---|---|---|
| Uniform | Round-Robin | 134.8 | 0.782 |
| Uniform | Prop-Fair | 162.8 | 0.641 |
| Uniform | **DQN** | **170.1** | 0.455 |
| Heterogeneous | Round-Robin | 175.8 | 0.885 |
| Heterogeneous | Prop-Fair | 199.2 | 0.745 |
| Heterogeneous | **DQN** | **206.5** | 0.587 |
| Bursty | Round-Robin | 163.3 | 0.793 |
| Bursty | Prop-Fair | 199.6 | 0.671 |
| Bursty | **DQN** | **201.5** | 0.585 |

### What each plot shows

**Result 1 — Throughput comparison**: DQN leads in all 9 scheduler–scenario combinations.

**Result 2 — Training reward curve**: all three curves converge above the random-action baseline within 20,000 steps. Heterogeneous converges fastest; bursty remains most volatile due to irreducible stochasticity from burst events.

**Result 3 — PRB allocation heatmap**: visualises DQN's learned policy in the heterogeneous scenario over 100 timesteps. UEs 0–1 (high CQI tier) are consistently dark; UE 4 (low CQI tier) stays at the 2-PRB floor. Sharp localised spikes confirm the agent reacts to transient CQI events.

**Result 4 — Jain's Fairness Index**: confirms the ordering Round-Robin > Prop-Fair > DQN across all scenarios. The DQN fairness deficit is a direct consequence of the reward structure — concentrating PRBs on high-CQI UEs dominates the log-utility term even with α = 0.9 weighting on fairness.

**Result 5 (comparison notebook) — Pareto scatter**: plots all nine scheduler–scenario combinations on a throughput vs fairness plane. No scheduler dominates on both axes simultaneously — the trade-off is intrinsic to the scheduling problem.

---

## Limitations and Future Work

- **Fairness deficit**: DQN scores lower on Jain's Index than both baselines despite the 0.9 PF reward weighting. A hard per-UE minimum throughput constraint in the reward function could close this gap.
- **Template quantisation**: the optimal allocation for a given CQI vector may not be exactly representable by any of the 64 templates. Increasing K reduces quantisation error at the cost of a larger action space.
- **Sim-to-real gap**: the environment uses a simplified Shannon model and does not capture inter-cell interference, UE mobility, HARQ retransmissions, or sub-PRB granularity.
- **Single-cell**: extension to multi-cell environments with inter-cell interference coordination is an important next step.
- **Future directions**: constrained RL, multi-objective reward shaping, multi-agent formulations, and evaluation under 3GPP-compliant channel models.

---

## Team

| Name | UNI | Contribution |
|---|---|---|
| Aarush Agarwal | aa5763 | Environment design, traffic scenarios|
| Arka Khorchidian | avk2137 | DQN agent, replay buffer, training loop |
| Keerthilakshmi Sivakumar | ks4420 | Baselines, metrics, evaluation notebooks |

---

## References

1. O-RAN Alliance, "O-RAN Architecture Description," Technical Specification v07.00, Feb. 2022.
2. A. Jalali et al., "Data throughput of CDMA-HDR," IEEE VTC Spring, 2000.
3. V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015.
4. H. van Hasselt et al., "Deep reinforcement learning with double Q-learning," AAAI, 2016.
5. Z. Wang et al., "Dueling network architectures for deep reinforcement learning," ICML, 2016.
6. M. Polese et al., "Understanding O-RAN," IEEE Communications Surveys & Tutorials, 2023.
7. R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, MIT Press, 2018.


