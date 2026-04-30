"""
eval_and_plot.py — Evaluate saved DQN models and regenerate all result figures.

Usage:
    python eval_and_plot.py
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))

from env.ran_environment import RANEnvironment
from agent.dqn_agent     import DQNAgent
from eval.baselines       import round_robin, proportional_fair
from eval.metrics         import compute_throughput, jains_fairness

matplotlib.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f8f8',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.size':        11,
})

SCENARIOS    = ['uniform', 'heterogeneous', 'bursty']
SCHEDULERS   = ['Round-Robin', 'Prop-Fair', 'DQN']
N_UES        = 5
TOTAL_PRBS   = 50
EVAL_STEPS   = 500
MODEL_DIR    = 'models'
BAR_COLORS   = ['#4C72B0', '#DD8452', '#55A868']
SC_MARKERS   = {'uniform': 'o', 'heterogeneous': 's', 'bursty': '^'}
SCHED_COLORS = {'Round-Robin': '#4C72B0', 'Prop-Fair': '#DD8452', 'DQN': '#55A868'}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_scheduler(scenario, scheduler_name, steps=EVAL_STEPS, seed=42):
    env = RANEnvironment(scenario=scenario, max_steps=steps)
    state, _ = env.reset(seed=seed)

    agent = None
    if scheduler_name == 'DQN':
        agent = DQNAgent()
        model_path = os.path.join(MODEL_DIR, f'dqn_trained_{scenario}.pt')
        if os.path.exists(model_path):
            agent.load(model_path)
        else:
            print(f'  [warn] No model at {model_path}; using untrained weights')

    all_throughputs, all_fairness, all_prb_alloc, all_cqi = [], [], [], []

    for _ in range(steps):
        if scheduler_name == 'Round-Robin':
            action = np.array(round_robin(N_UES, TOTAL_PRBS), dtype=np.float32)
        elif scheduler_name == 'Prop-Fair':
            action = proportional_fair(state, N_UES, TOTAL_PRBS)
        else:
            _, action = agent.act(state, epsilon=0.0)

        state, _, terminated, truncated, info = env.step(action)
        tput = np.array(info['throughputs'])
        all_throughputs.append(tput)
        all_fairness.append(jains_fairness(tput))
        all_prb_alloc.append(env.prb_alloc.copy())
        all_cqi.append(np.array(info['cqi']))

        if terminated or truncated:
            state, _ = env.reset(seed=seed)

    return {
        'throughputs': np.array(all_throughputs),
        'fairness':    np.array(all_fairness),
        'prb_alloc':   np.array(all_prb_alloc),
        'cqi':         np.array(all_cqi),
    }


print('Running evaluations (9 combinations)...')
results = {}
for scenario in SCENARIOS:
    results[scenario] = {}
    for sched in SCHEDULERS:
        print(f'  {scenario:>14s} x {sched}')
        results[scenario][sched] = evaluate_scheduler(scenario, sched)
print('Done.\n')


# ── Summary table ─────────────────────────────────────────────────────────────

col_w = [16, 14, 16, 13]
print('=' * 63)
print('SUMMARY')
print('=' * 63)
print(f"{'Scenario':>{col_w[0]}} {'Scheduler':>{col_w[1]}} {'Avg Total Tput':>{col_w[2]}} {'Jains Index':>{col_w[3]}}")
print('-' * 63)
for sc in SCENARIOS:
    for sched in SCHEDULERS:
        tput = results[sc][sched]['throughputs'].sum(axis=1).mean()
        jain = results[sc][sched]['fairness'].mean()
        print(f'{sc:>{col_w[0]}} {sched:>{col_w[1]}} {tput:>{col_w[2]}.2f} {jain:>{col_w[3]}.4f}')
    print('-' * 63)
print()


# ── Result 1: Throughput bar chart ────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle('Result 1: Average Throughput — 3 Schedulers x 3 Scenarios',
             fontsize=14, fontweight='bold', y=1.02)

x     = np.arange(N_UES)
width = 0.25

for col, scenario in enumerate(SCENARIOS):
    ax = axes[col]
    for i, sched in enumerate(SCHEDULERS):
        avg_tput = results[scenario][sched]['throughputs'].mean(axis=0)
        ax.bar(x + (i - 1) * width, avg_tput, width,
               label=sched, color=BAR_COLORS[i], edgecolor='white', linewidth=0.6)

    ax.set_title(f'{scenario.capitalize()} Scenario', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'UE {i}' for i in range(N_UES)])
    ax.set_ylabel('Avg Throughput (a.u.)')
    ax.legend(fontsize=9)

    totals = [results[scenario][s]['throughputs'].mean(axis=0).sum() for s in SCHEDULERS]
    total_str = '  '.join(f'{s}: {t:.1f}' for s, t in zip(SCHEDULERS, totals))
    ax.set_xlabel(f'Total -> {total_str}', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'result1_throughput.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved result1_throughput.png')


# ── Result 2: Training curve (load from saved npz) ────────────────────────────

CURVES_PATH = os.path.join(MODEL_DIR, 'training_curves.npz')
sc_colors   = {'uniform': '#4C72B0', 'heterogeneous': '#DD8452', 'bursty': '#55A868'}

def smooth(y, window=50):
    return np.convolve(y, np.ones(window) / window, mode='valid') if len(y) >= window else y

if os.path.exists(CURVES_PATH):
    data   = np.load(CURVES_PATH)
    curves = {sc: (data[f'{sc}_steps'], data[f'{sc}_rewards']) for sc in SCENARIOS}

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle('Result 2: DQN Training Reward Curve (PF reward)',
                 fontsize=14, fontweight='bold')

    all_sm = []
    for sc in SCENARIOS:
        steps, rewards = curves[sc]
        sm = smooth(rewards, window=50)
        valid_steps = steps[49:]
        ax.plot(steps, rewards, alpha=0.08, color=sc_colors[sc], linewidth=0.5)
        ax.plot(valid_steps, sm, label=f'{sc} (smoothed)',
                color=sc_colors[sc], linewidth=2.5)
        all_sm.append(sm)

    all_vals = np.concatenate(all_sm)
    ymin, ymax = all_vals.min(), all_vals.max()
    margin = (ymax - ymin) * 0.25
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Avg Step Reward')
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'result2_training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved result2_training_curve.png')
else:
    print('Skipped result2_training_curve.png (no training_curves.npz found)')


# ── Result 3: PRB allocation heatmap ─────────────────────────────────────────

HEATMAP_STEPS = 100

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle(
    'Result 3: PRB Allocation Heatmap — Heterogeneous Scenario\n'
    '(rows = UEs; columns = time steps; colour = PRBs allocated)',
    fontsize=12, fontweight='bold'
)

# Use a shared color range derived from actual data so pale schedulers aren't invisible
all_prb_data = np.concatenate([
    results['heterogeneous'][s]['prb_alloc'][:HEATMAP_STEPS] for s in SCHEDULERS
])
vmin_shared = 0
vmax_shared = all_prb_data.max()

for row, sched in enumerate(SCHEDULERS):
    ax   = axes[row]
    data = results['heterogeneous'][sched]['prb_alloc'][:HEATMAP_STEPS].T
    im   = ax.imshow(data, aspect='auto', cmap='YlOrRd',
                     vmin=vmin_shared, vmax=vmax_shared, interpolation='nearest')
    ax.set_yticks(range(N_UES))
    ax.set_yticklabels([f'UE {i}' for i in range(N_UES)], fontsize=9)
    ax.set_xlabel('Time Step')
    ax.set_title(sched, fontweight='bold', fontsize=11)
    plt.colorbar(im, ax=ax, label='PRBs', fraction=0.015, pad=0.01)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(MODEL_DIR, 'result3_prb_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved result3_prb_heatmap.png')


# ── Result 4: Jain's Fairness Index bar chart ─────────────────────────────────

jain_matrix = np.zeros((len(SCENARIOS), len(SCHEDULERS)))
for i, sc in enumerate(SCENARIOS):
    for j, sched in enumerate(SCHEDULERS):
        jain_matrix[i, j] = results[sc][sched]['fairness'].mean()

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Result 4: Jain's Fairness Index — All Schedulers x All Scenarios",
             fontsize=13, fontweight='bold')

x     = np.arange(len(SCENARIOS))
width = 0.22
for j, (sched, color) in enumerate(zip(SCHEDULERS, BAR_COLORS)):
    vals = jain_matrix[:, j]
    bars = ax.bar(x + (j - 1) * width, vals, width,
                  label=sched, color=color, edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in SCENARIOS])
ax.set_ylabel("Jain's Fairness Index")
ax.set_ylim(0, 1.08)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='Perfect fairness')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'result4_jains_fairness.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved result4_jains_fairness.png')


# ── Result 5: Throughput–Fairness scatter ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Result 5: Throughput–Fairness Trade-off\n'
             '(each point = scheduler × scenario)', fontsize=13, fontweight='bold')

for sched in SCHEDULERS:
    for sc in SCENARIOS:
        tput = results[sc][sched]['throughputs'].sum(axis=1).mean()
        jain = results[sc][sched]['fairness'].mean()
        ax.scatter(tput, jain,
                   color=SCHED_COLORS[sched],
                   marker=SC_MARKERS[sc],
                   s=120, zorder=3,
                   edgecolors='white', linewidths=0.8)
        ax.annotate(f'{sc[:3].title()}',
                    (tput, jain), textcoords='offset points',
                    xytext=(6, 3), fontsize=7.5, color='#444')

for sched, color in SCHED_COLORS.items():
    ax.scatter([], [], color=color, label=sched, s=80)
for sc, marker in SC_MARKERS.items():
    ax.scatter([], [], color='gray', marker=marker,
               label={'uniform': 'Uniform', 'heterogeneous': 'Heterogeneous', 'bursty': 'Bursty'}[sc], s=80)

ax.set_xlabel('Avg Total Throughput (a.u.)', fontsize=11)
ax.set_ylabel("Jain's Fairness Index", fontsize=11)
ax.set_ylim(0.3, 1.05)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.legend(fontsize=9, loc='lower right', ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'result5_throughput_fairness.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved result5_throughput_fairness.png')

print('\nAll figures updated in models/')
