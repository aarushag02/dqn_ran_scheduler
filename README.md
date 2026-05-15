# DQN-based Dynamic Spectrum Sharing for O-RAN near-RT RIC

A Deep Q-Network (DQN) agent for dynamic Physical Resource Block (PRB)
allocation in a simulated O-RAN near Real-Time RAN Intelligent Controller
(near-RT RIC) environment. The agent is trained to maximise a Proportional
Fair utility objective, balancing aggregate throughput against per-UE fairness,
and is evaluated against round-robin and proportional-fair baseline schedulers
across three traffic scenarios.

## Results Summary

| Scenario | Scheduler | Avg Total Throughput | Jain's Fairness |
|---|---|---|---|
| Uniform | Round-Robin | 134.8 | 0.782 |
| Uniform | Prop-Fair | 162.8 | 0.641 |
| **Uniform** | **DQN** | **142.4** | **0.709** |
| Heterogeneous | Round-Robin | 175.8 | 0.886 |
| Heterogeneous | Prop-Fair | 199.2 | 0.745 |
| **Heterogeneous** | **DQN** | **178.9** | **0.860** |
| Bursty | Round-Robin | 163.3 | 0.794 |
| Bursty | Prop-Fair | 199.6 | 0.671 |
| **Bursty** | **DQN** | **171.0** | **0.719** |

The DQN consistently achieves the best fairness-throughput tradeoff among all RL methods — outperforming Prop-Fair on fairness in every scenario while maintaining higher throughput than Round-Robin.

## Project Structure

```
dqn_ran_scheduler/
├── env/
│   └── ran_environment.py      # Custom Gymnasium environment
├── agent/
│   ├── dqn_agent.py            # Dueling Double DQN agent
│   └── replay_buffer.py        # Prioritized Experience Replay + NStepBuffer
├── eval/
│   ├── baselines.py            # Round-robin & proportional-fair schedulers
│   ├── metrics.py              # Throughput & Jain's fairness index
│   ├── Baselines_2.py          # Extended baselines (max-weight)
│   └── Metrics_2.py            # Extended metrics (latency, energy efficiency)
├── models/
│   ├── dqn_trained_*.pt        # Saved DQN weights (3 scenarios)
│   ├── ppo_trained_*.zip       # Saved PPO weights (3 scenarios)
│   ├── ddpg_trained_*.zip      # Saved DDPG weights (3 scenarios)
│   ├── result1_throughput.png
│   ├── result2_training_curve.png
│   ├── result3_prb_heatmap.png
│   ├── result4_jains_fairness.png
│   ├── result5_throughput_fairness.png
│   ├── comp_throughput.png
│   ├── comp_fairness.png
│   ├── comp_training_curves.png
│   └── comp_per_ue.png
├── notebooks/
│   ├── results.ipynb           # DQN vs baselines result plots
│   └── comparison.ipynb        # DQN vs PPO vs DDPG comparison plots
├── train.py                    # DQN training loop
├── train_sb3.py                # PPO / DDPG training (Stable-Baselines3)
├── eval_and_plot.py            # Standalone evaluation + result figure generation
├── eval_comparison.py          # Standalone comparison figure generation
├── sweep_alpha.py              # Pareto sweep over PF/throughput blend weight
├── demo.py                     # Live animation demo
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training

### DQN

```bash
# Train on all three scenarios (100k steps each)
python train.py --scenario uniform
python train.py --scenario heterogeneous
python train.py --scenario bursty

# Custom options
python train.py --scenario bursty --total-steps 200000 --lr 1e-4
```

Training prints step, epsilon, average reward, and loss every 1000 steps.
Model saved to `models/dqn_trained_<scenario>.pt`.

### PPO / DDPG (Stable-Baselines3)

```bash
python train_sb3.py --algo ppo  --scenario uniform
python train_sb3.py --algo ddpg --scenario heterogeneous --total-steps 100000
```

Models saved as `models/<algo>_trained_<scenario>.zip`.
Reward histories saved as `models/<algo>_rewards_<scenario>.npy`.

## Evaluation & Plots

### DQN vs baselines (5 result figures)

```bash
python eval_and_plot.py
```

Loads saved DQN `.pt` models, evaluates all 9 combinations (3 schedulers × 3 scenarios),
and saves to `models/`:

- `result1_throughput.png` — per-UE average throughput
- `result2_training_curve.png` — DQN training reward curve
- `result3_prb_heatmap.png` — PRB allocation heatmap (heterogeneous)
- `result4_jains_fairness.png` — Jain's fairness index comparison
- `result5_throughput_fairness.png` — throughput-fairness Pareto scatter

### DQN vs PPO vs DDPG (4 comparison figures)

```bash
python eval_comparison.py
```

Evaluates all 15 combinations (5 methods × 3 scenarios) and saves to `models/`:

- `comp_throughput.png` — total throughput comparison
- `comp_training_curves.png` — training reward curves per scenario
- `comp_fairness.png` — Jain's fairness index comparison
- `comp_per_ue.png` — per-UE throughput (heterogeneous scenario)

### Alpha sweep (fairness-throughput Pareto analysis)

```bash
python sweep_alpha.py
```

Trains lightweight proxy models across blend weights α ∈ [0, 1] and plots the
throughput-fairness Pareto curve to identify the optimal α before full retraining.
Saves `models/sweep_alpha_pareto.png`.

## Live Demo

```bash
# Animate DQN policy (loads models/dqn_trained_uniform.pt)
python demo.py

# Bursty scenario with side-by-side baseline comparison
python demo.py --scenario bursty --compare-baselines

# Run without a trained model (random untrained policy)
python demo.py --no-model
```

## Architecture

### Environment (`env/ran_environment.py`)

| Parameter | Value |
|---|---|
| UEs | 5 |
| Total PRBs | 50 |
| Min PRBs per UE | 2 (hard floor) |
| Episode length | 200 steps |
| State dim | 15 floats (see below) |
| Action | Index into 64 pre-built PRB allocation templates |
| Reward | 0.9 × PF utility + 0.1 × normalised throughput |

**State vector** (CQI-rank sorted, descending):

| Indices | Feature | Range |
|---|---|---|
| 0–4 | Normalised CQI: `(cqi − 1) / 14` | [0, 1] |
| 5–9 | Normalised PRB alloc: `prb / 50` | [0, 1] |
| 10–14 | Relative throughput deficit: `(ema_i − mean_ema) / mean_ema` | ~[−2, 2] |

The deficit feature gives the agent a fairness memory — it can see which UEs have
been systematically under-served and compensate dynamically.

**Reward function:**

```
PF_utility  = mean_i( log(1 + throughput_i) )   # proportional fair
linear      = sum(throughputs) / total_prbs       # normalised aggregate
reward      = 0.9 × PF_utility + 0.1 × linear
```

The blend weight α=0.9 was selected via Pareto sweep (`sweep_alpha.py`) as the
point that maximises throughput gain over Round-Robin while keeping fairness as
the dominant objective.

**Three traffic scenarios:**

| Scenario | CQI initialisation | Evolution |
|---|---|---|
| Uniform | Normal(8, 2), clamped [1,15] | Gaussian drift σ=0.5 |
| Heterogeneous | Tiered: high [10,15], medium [5,9], low [1,4] | Drift + mean-reversion to tier |
| Bursty | Normal(8, 2) | Drift + 10% spike probability ±[4,8] |

### DQN Agent (`agent/dqn_agent.py`)

**Architecture — Dueling DQN:**

```
Input (15)
  └─ Backbone: Linear(15→256) → ReLU → Linear(256→256) → ReLU
       ├─ Value stream:     Linear(256→128) → ReLU → Linear(128→1)
       └─ Advantage stream: Linear(256→128) → ReLU → Linear(128→64)
  Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a))
```

**Action space — 64 fixed PRB allocation templates:**

| Group | Count | Description |
|---|---|---|
| Equal split | 1 | 10 PRBs per UE |
| Single-UE heavy | 5 | One UE gets 40%, rest share 60% |
| Single-UE moderate | 5 | One UE gets 30%, rest share 70% |
| Two-UE heavy | 10 | Pair shares 64%, rest share 36% |
| Two-UE moderate | 10 | Pair shares 48%, rest share 52% |
| Random Dirichlet | 33 | Varied concentration for exploration diversity |

**Training hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam, lr=1e-4 |
| Loss | Huber (IS-weighted for PER) |
| Target network update | Hard copy every 200 gradient steps |
| Exploration | ε-greedy, ε: 1.0 → 0.05 (decay 0.9997) |
| Gradient clipping | max norm 1.0 |
| Discount factor (γ) | 0.95 |
| N-step returns | 5 |
| Replay buffer | Prioritized Experience Replay (α=0.6, β: 0.4→1.0) |
| Buffer capacity | 50,000 |
| Batch size | 64 |
| Warmup steps | 1,000 |
| Total training steps | 100,000 per scenario |

**Learning algorithm — Double DQN:**

```
a* = argmax_a Q_online(s', a)     # online net selects action
y  = r_n + γ^n · Q_target(s', a*) # target net evaluates it
loss = IS_weight · Huber(Q_online(s, a), y)
```

### Replay Buffer (`agent/replay_buffer.py`)

**`PrioritizedReplayBuffer`** — Schaul et al. 2016. Samples experiences proportional
to `|TD error|^α`, giving more training weight to surprising/high-error transitions
(e.g. successfully recovering a starved UE). Importance-sampling weights correct for
the introduced bias; β is annealed from 0.4 → 1.0 over training.

**`NStepBuffer`** — accumulates n=5 consecutive transitions and emits a single
(s₀, a₀, G₅, s₅, done) tuple where G₅ is the 5-step discounted return. Handles
episode boundaries correctly.
