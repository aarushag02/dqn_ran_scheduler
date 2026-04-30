"""
sweep_alpha.py — Sweep the PF/throughput blend weight α to find the
                 optimal fairness-throughput tradeoff before full retraining.

Reward = α * PF_utility + (1 - α) * linear_throughput

Trains on the uniform scenario for 40k steps per α value (fast proxy).
Evaluates Jain's fairness and total throughput, then plots the Pareto curve.

Usage:
    python sweep_alpha.py
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
sys.path.insert(0, os.path.dirname(__file__))

from env.ran_environment  import RANEnvironment
from agent.dqn_agent      import DQNAgent
from agent.replay_buffer  import ReplayBuffer, NStepBuffer
from eval.baselines        import round_robin, proportional_fair
from eval.metrics          import jains_fairness

matplotlib.rcParams.update({'figure.facecolor': 'white', 'font.size': 11})

# ── Config ────────────────────────────────────────────────────────────────────

ALPHAS       = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SCENARIO     = "uniform"       # proxy scenario for the sweep
TRAIN_STEPS  = 40_000
EVAL_STEPS   = 500
N_UES        = 5
TOTAL_PRBS   = 50
MODEL_DIR    = "models"


# ── Patched reward that accepts α ─────────────────────────────────────────────

def blended_reward(throughputs, n_ues, total_prbs, alpha):
    pf_utility = np.sum(np.log1p(throughputs)) / n_ues
    linear     = throughputs.sum() / total_prbs
    return alpha * pf_utility + (1.0 - alpha) * linear


# ── Training ──────────────────────────────────────────────────────────────────

def train_alpha(alpha, scenario=SCENARIO, total_steps=TRAIN_STEPS):
    env       = RANEnvironment(scenario=scenario)
    agent     = DQNAgent(lr=1e-4, gamma=0.95, target_update=200, n_step=3)
    buffer    = ReplayBuffer(50_000)
    nstep_buf = NStepBuffer(n=3, gamma=0.95)

    eps, eps_min, eps_decay = 1.0, 0.05, 0.9997
    state, _ = env.reset(seed=0)

    for t in range(1, total_steps + 1):
        idx, alloc = agent.act(state, eps)
        ns, _, term, trunc, info = env.step(alloc)
        done = term or trunc

        # Override the env reward with our blended version
        r = blended_reward(np.array(info['throughputs']), N_UES, TOTAL_PRBS, alpha)

        tr = nstep_buf.push(state, idx, r, ns, done)
        if tr is not None:
            buffer.push(*tr)

        eps = max(eps_min, eps * eps_decay)
        if t >= 1000 and len(buffer) >= 64:
            agent.learn(buffer.sample(64))

        if done:
            for tr in nstep_buf.flush():
                buffer.push(*tr)
            nstep_buf.clear()
            state, _ = env.reset()
        else:
            state = ns

    return agent


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(agent, scenario=SCENARIO, steps=EVAL_STEPS):
    env = RANEnvironment(scenario=scenario, max_steps=steps)
    state, _ = env.reset(seed=42)
    throughputs, fairness = [], []

    for _ in range(steps):
        _, action = agent.act(state, epsilon=0.0)
        state, _, term, trunc, info = env.step(action)
        tput = np.array(info['throughputs'])
        throughputs.append(tput)
        fairness.append(jains_fairness(tput))
        if term or trunc:
            state, _ = env.reset(seed=42)

    return np.array(throughputs).sum(axis=1).mean(), np.mean(fairness)


def baseline_eval(scheduler_name, scenario=SCENARIO, steps=EVAL_STEPS):
    env = RANEnvironment(scenario=scenario, max_steps=steps)
    state, _ = env.reset(seed=42)
    throughputs, fairness = [], []

    for _ in range(steps):
        if scheduler_name == 'RR':
            action = np.array(round_robin(N_UES, TOTAL_PRBS), dtype=np.float32)
        else:
            action = proportional_fair(state, N_UES, TOTAL_PRBS)
        state, _, term, trunc, info = env.step(action)
        tput = np.array(info['throughputs'])
        throughputs.append(tput)
        fairness.append(jains_fairness(tput))
        if term or trunc:
            state, _ = env.reset(seed=42)

    return np.array(throughputs).sum(axis=1).mean(), np.mean(fairness)


# ── Main sweep ────────────────────────────────────────────────────────────────

print(f"Sweeping α over {ALPHAS} | scenario={SCENARIO} | train_steps={TRAIN_STEPS}")
print("-" * 60)

sweep_results = {}
for alpha in ALPHAS:
    print(f"  α={alpha:.1f} — training...", end=" ", flush=True)
    agent = train_alpha(alpha)
    tput, jain = evaluate(agent)
    sweep_results[alpha] = (tput, jain)
    print(f"throughput={tput:.2f}  Jain={jain:.4f}")

rr_tput, rr_jain = baseline_eval('RR')
pf_tput, pf_jain = baseline_eval('PF')

print("\nBaselines:")
print(f"  Round-Robin : throughput={rr_tput:.2f}  Jain={rr_jain:.4f}")
print(f"  Prop-Fair   : throughput={pf_tput:.2f}  Jain={pf_jain:.4f}")


# ── Find optimal α ────────────────────────────────────────────────────────────
# Nash bargaining: maximise product of gains over the RR reference point

best_alpha, best_score = None, -np.inf
print("\nNash bargaining scores (ref = Round-Robin):")
for alpha, (tput, jain) in sweep_results.items():
    tput_gain = max(0.0, tput - rr_tput)
    jain_gain = max(0.0, jain - rr_jain)
    score = tput_gain * jain_gain
    print(f"  α={alpha:.1f}  tput_gain={tput_gain:+.2f}  jain_gain={jain_gain:+.4f}  score={score:.4f}")
    if score > best_score:
        best_score = score
        best_alpha = alpha

print(f"\n→ Optimal α = {best_alpha}  (Nash bargaining score = {best_score:.4f})")


# ── Plot ──────────────────────────────────────────────────────────────────────

alphas  = list(sweep_results.keys())
tputs   = [sweep_results[a][0] for a in alphas]
jains   = [sweep_results[a][1] for a in alphas]

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'α Sweep: Throughput–Fairness Pareto Curve\n'
             f'(scenario={SCENARIO}, {TRAIN_STEPS//1000}k train steps per α)',
             fontsize=13, fontweight='bold')

# DQN sweep curve
sc = ax.scatter(tputs, jains, c=alphas, cmap='plasma', s=120,
                zorder=4, edgecolors='white', linewidths=0.8)
plt.colorbar(sc, ax=ax, label='α (PF weight)')

for a, t, j in zip(alphas, tputs, jains):
    ax.annotate(f'α={a}', (t, j), textcoords='offset points',
                xytext=(6, 3), fontsize=8)

# Connect sweep points with a line to show the Pareto frontier
order = np.argsort(tputs)
ax.plot(np.array(tputs)[order], np.array(jains)[order],
        color='gray', linewidth=1, linestyle='--', alpha=0.5, zorder=2)

# Baselines
ax.scatter(rr_tput, rr_jain, marker='*', s=250, color='#4C72B0',
           zorder=5, label='Round-Robin', edgecolors='white')
ax.scatter(pf_tput, pf_jain, marker='*', s=250, color='#DD8452',
           zorder=5, label='Prop-Fair', edgecolors='white')

# Highlight optimal
opt_tput, opt_jain = sweep_results[best_alpha]
ax.scatter(opt_tput, opt_jain, marker='D', s=180, color='lime',
           zorder=6, label=f'Optimal α={best_alpha}', edgecolors='black', linewidths=1)

ax.set_xlabel('Avg Total Throughput (a.u.)', fontsize=11)
ax.set_ylabel("Jain's Fairness Index", fontsize=11)
ax.axhline(rr_jain, color='#4C72B0', linestyle=':', linewidth=1, alpha=0.6)
ax.axvline(rr_tput, color='#4C72B0', linestyle=':', linewidth=1, alpha=0.6)
ax.legend(fontsize=9)

out_path = os.path.join(MODEL_DIR, 'sweep_alpha_pareto.png')
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPareto curve saved to {out_path}")
