"""
train.py — DQN training loop for the O-RAN PRB scheduler.

Usage
-----
    python train.py                        # uniform scenario (default)
    python train.py --scenario bursty
    python train.py --scenario heterogeneous --total-steps 50000
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.ran_environment import RANEnvironment
from agent.dqn_agent     import DQNAgent
from agent.replay_buffer import PrioritizedReplayBuffer, NStepBuffer


# ------------------------------------------------------------------
# Hyper-parameters
# ------------------------------------------------------------------
DEFAULTS = dict(
    scenario    = "uniform",
    total_steps = 100_000,
    batch_size  = 64,
    buffer_cap  = 50_000,
    lr          = 1e-4,
    gamma       = 0.95,
    eps_start   = 1.0,
    eps_end     = 0.05,
    eps_decay   = 0.9997,
    target_upd  = 200,
    warmup      = 1_000,
    n_step      = 5,
    log_every   = 1_000,
    model_dir   = "models",
)


def parse_args():
    p = argparse.ArgumentParser(description="Train DQN PRB scheduler")
    p.add_argument("--scenario",     default=DEFAULTS["scenario"],
                   choices=["uniform", "heterogeneous", "bursty"])
    p.add_argument("--total-steps",  type=int,   default=DEFAULTS["total_steps"])
    p.add_argument("--batch-size",   type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    p.add_argument("--gamma",        type=float, default=DEFAULTS["gamma"])
    p.add_argument("--model-dir",    default=DEFAULTS["model_dir"])
    return p.parse_args()


def train(cfg):
    os.makedirs(cfg.model_dir, exist_ok=True)

    env       = RANEnvironment(scenario=cfg.scenario)
    agent     = DQNAgent(lr=cfg.lr, gamma=cfg.gamma,
                         target_update=DEFAULTS["target_upd"],
                         n_step=DEFAULTS["n_step"])
    buffer    = PrioritizedReplayBuffer(capacity=DEFAULTS["buffer_cap"],
                                        beta_steps=cfg.total_steps)
    nstep_buf = NStepBuffer(n=DEFAULTS["n_step"], gamma=cfg.gamma)

    epsilon     = DEFAULTS["eps_start"]
    total_steps = 0
    ep_reward   = 0.0
    recent_losses  = []
    recent_rewards = []

    state, _ = env.reset()

    print(f"Training DQN | scenario={cfg.scenario} | "
          f"total_steps={cfg.total_steps} | n_step={DEFAULTS['n_step']} | "
          f"n_templates={agent.n_templates} | device={agent.device}")
    print("-" * 65)

    while total_steps < cfg.total_steps:

        action_idx, prb_alloc = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(prb_alloc)
        done = terminated or truncated

        # accumulate n-step return; push to replay when ready
        t = nstep_buf.push(state, action_idx, reward, next_state, done)
        if t is not None:
            buffer.push(*t)

        ep_reward  += reward
        total_steps += 1
        epsilon = max(DEFAULTS["eps_end"], epsilon * DEFAULTS["eps_decay"])

        if total_steps >= DEFAULTS["warmup"] and len(buffer) >= cfg.batch_size:
            batch              = buffer.sample(cfg.batch_size)
            loss, td_errors    = agent.learn(batch)
            buffer.update_priorities(batch[6], td_errors)
            recent_losses.append(loss)

        recent_rewards.append(reward)

        if total_steps % DEFAULTS["log_every"] == 0:
            avg_r = np.mean(recent_rewards[-DEFAULTS["log_every"]:])
            avg_l = (np.mean(recent_losses[-DEFAULTS["log_every"]:])
                     if recent_losses else float("nan"))
            print(f"step {total_steps:>6d} | eps {epsilon:.3f} | "
                  f"avg_reward {avg_r:>7.3f} | loss {avg_l:.4f}")

        if done:
            for t in nstep_buf.flush():
                buffer.push(*t)
            nstep_buf.clear()
            state, _ = env.reset()
            ep_reward = 0.0
        else:
            state = next_state

    model_path = os.path.join(cfg.model_dir, f"dqn_trained_{cfg.scenario}.pt")
    agent.save(model_path)
    print("-" * 65)
    print(f"Training complete. Model saved to {model_path}")
    return agent


if __name__ == "__main__":
    args = parse_args()
    train(args)
