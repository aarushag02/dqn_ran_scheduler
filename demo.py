"""
demo.py — Live animation demo for the DQN PRB scheduler.

Runs the trained DQN policy (and optionally the two baselines) side-by-side
and animates PRB allocation, CQI, and per-UE throughput in real time.
Intended for use during the project presentation.

Usage
    python demo.py                          # uniform scenario, DQN only
    python demo.py - scenario bursty        # bursty scenario
    python demo.py - scenario heterogeneous - compare-baselines
    python demo.py - no-model               # random policy (no .pt needed)

Controls
    Close the window to exit early.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(__file__))

from env.ran_environment import RANEnvironment
from agent.dqn_agent     import DQNAgent

# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Live DQN PRB scheduler demo")
    p.add_argument("--scenario", default="uniform",
                   choices=["uniform", "heterogeneous", "bursty"])
    p.add_argument("--model-dir",   default="models")
    p.add_argument("--no-model",    action="store_true",
                   help="Skip loading a trained model; use random policy")
    p.add_argument("--compare-baselines", action="store_true",
                   help="Show round-robin and proportional-fair alongside DQN")
    p.add_argument("--steps",       type=int, default=200,
                   help="Number of steps to animate per episode")
    p.add_argument("--interval",    type=int, default=150,
                   help="Milliseconds between animation frames")
    return p.parse_args()

# Baseline helpers (inline to avoid import dependency on eval/)
def round_robin_alloc(n_ues, total_prbs):
    return np.full(n_ues, total_prbs / n_ues, dtype=np.float32)

def proportional_fair_alloc(state, n_ues, total_prbs):
    cqi = state[:n_ues].astype(np.float64)
    cqi = np.clip(cqi, 1e-6, None)
    weights = cqi / cqi.sum()
    return (weights * total_prbs).astype(np.float32)

# Animation
N_UES     = 5
TOTAL_PRB = 50
UE_LABELS = [f"UE {i}" for i in range(N_UES)]
COLORS    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def build_figure(compare: bool):
    """Create the Matplotlib figure layout."""
    rows = 3 if not compare else 4
    fig = plt.figure(figsize=(13, 8), facecolor="#1e1e2e")
    fig.suptitle("DQN O-RAN PRB Scheduler — Live Demo",
                 color="white", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(rows, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.07)

    axes = {}

    # row 0: PRB allocation bar chart
    axes["prb"] = fig.add_subplot(gs[0, :])

    # row 1: CQI per UE (line) + reward over time (line)
    axes["cqi"]    = fig.add_subplot(gs[1, 0])
    axes["reward"] = fig.add_subplot(gs[1, 1])

    # row 2: per-UE throughput bar
    axes["tput"] = fig.add_subplot(gs[2, :])

    if compare:
        axes["compare"] = fig.add_subplot(gs[3, :])

    for ax in axes.values():
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    return fig, axes


class LiveDemo:
    def __init__(self, args):
        self.args    = args
        self.env     = RANEnvironment(scenario=args.scenario,
                                      max_steps=args.steps)
        self.agent   = DQNAgent()

        if not args.no_model:
            model_path = os.path.join(args.model_dir,
                                      f"dqn_trained_{args.scenario}.pt")
            if os.path.exists(model_path):
                self.agent.load(model_path)
                print(f"Loaded model: {model_path}")
            else:
                print(f"[warn] Model not found at {model_path}. "
                      "Using untrained weights.")

        self.fig, self.axes = build_figure(args.compare_baselines)

        # history buffers
        self.reward_hist = []
        self.cqi_hist    = [[] for _ in range(N_UES)]
        self.step_nums   = []

        # episode state
        self.state, _ = self.env.reset()

    # per-frame update

    def update(self, frame):
        _, action  = self.agent.act(self.state, epsilon=0.0)
        state, reward, terminated, truncated, info = self.env.step(action)
        done       = terminated or truncated
        throughputs = np.array(info["throughputs"])
        cqi         = np.array(info["cqi"])
        prb_alloc   = self.env.prb_alloc

        # record history
        self.step_nums.append(frame)
        self.reward_hist.append(reward)
        for i in range(N_UES):
            self.cqi_hist[i].append(cqi[i])

        # PRB allocation bar
        ax = self.axes["prb"]
        ax.cla()
        ax.set_facecolor("#2a2a3e")
        bars = ax.bar(UE_LABELS, prb_alloc, color=COLORS, edgecolor="white",
                      linewidth=0.5)
        ax.set_ylim(0, TOTAL_PRB * 1.15)
        ax.set_title("PRB Allocation (DQN)", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_ylabel("PRBs", color="white", fontsize=8)
        for bar, v in zip(bars, prb_alloc):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.1f}", ha="center", va="bottom",
                    color="white", fontsize=7)
        ax.axhline(TOTAL_PRB / N_UES, color="#aaaacc", linestyle="--",
                   linewidth=0.8, label="equal split")
        ax.legend(fontsize=7, facecolor="#2a2a3e", labelcolor="white",
                  framealpha=0.6)

        # CQI history lines
        ax = self.axes["cqi"]
        ax.cla()
        ax.set_facecolor("#2a2a3e")
        for i in range(N_UES):
            ax.plot(self.step_nums, self.cqi_hist[i],
                    color=COLORS[i], label=UE_LABELS[i], linewidth=1.2)
        ax.set_ylim(0, 16)
        ax.set_title("CQI over Time", color="white", fontsize=10)
        ax.set_ylabel("CQI", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.legend(fontsize=6, facecolor="#2a2a3e", labelcolor="white",
                  ncol=5, loc="upper right", framealpha=0.6)

        # Reward history
        ax = self.axes["reward"]
        ax.cla()
        ax.set_facecolor("#2a2a3e")
        ax.plot(self.step_nums, self.reward_hist,
                color="#a8d8a8", linewidth=1.2)
        ax.set_title("Step Reward", color="white", fontsize=10)
        ax.set_ylabel("Reward", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)

        # running average overlay
        window = min(20, len(self.reward_hist))
        if window > 1:
            ra = np.convolve(self.reward_hist,
                             np.ones(window) / window, mode="valid")
            ax.plot(self.step_nums[window - 1:], ra,
                    color="#ffcc88", linewidth=1.5, linestyle="--",
                    label=f"{window}-step avg")
            ax.legend(fontsize=7, facecolor="#2a2a3e",
                      labelcolor="white", framealpha=0.6)

        # Throughput bars
        ax = self.axes["tput"]
        ax.cla()
        ax.set_facecolor("#2a2a3e")
        bars = ax.bar(UE_LABELS, throughputs, color=COLORS,
                      edgecolor="white", linewidth=0.5)
        ax.set_title("Throughput this Step (DQN)", color="white", fontsize=10)
        ax.set_ylabel("Throughput (a.u.)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        for bar, v in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                    f"{v:.1f}", ha="center", va="bottom",
                    color="white", fontsize=7)

        # Baseline comparison (optional)
        if self.args.compare_baselines and "compare" in self.axes:
            ax = self.axes["compare"]
            ax.cla()
            ax.set_facecolor("#2a2a3e")

            rr_alloc = round_robin_alloc(N_UES, TOTAL_PRB)
            pf_alloc = proportional_fair_alloc(self.state, N_UES, TOTAL_PRB)

            snr  = (cqi - 1) * 2.0
            rr_tput = rr_alloc * np.log2(1.0 + snr)
            pf_tput = pf_alloc * np.log2(1.0 + snr)

            x     = np.arange(N_UES)
            width = 0.25
            ax.bar(x - width, rr_tput,      width, label="Round-Robin",
                   color="#6699cc", edgecolor="white", linewidth=0.5)
            ax.bar(x,          pf_tput,      width, label="Prop-Fair",
                   color="#cc9966", edgecolor="white", linewidth=0.5)
            ax.bar(x + width,  throughputs,  width, label="DQN",
                   color="#66cc99", edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(UE_LABELS, color="white", fontsize=8)
            ax.set_title("Throughput Comparison: RR vs PF vs DQN",
                         color="white", fontsize=10)
            ax.set_ylabel("Throughput (a.u.)", color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=8)
            ax.legend(fontsize=8, facecolor="#2a2a3e",
                      labelcolor="white", framealpha=0.7)

        self.state = state

        if done:
            self.state, _ = self.env.reset()
            self.reward_hist.clear()
            self.cqi_hist = [[] for _ in range(N_UES)]
            self.step_nums.clear()

    # Run

    def run(self):
        anim = FuncAnimation(
            self.fig,
            self.update,
            interval=self.args.interval,
            cache_frame_data=False,
        )
        plt.show()

# Entry point
if __name__ == "__main__":
    args = parse_args()
    demo = LiveDemo(args)
    demo.run()
