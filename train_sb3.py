"""
train_sb3.py — Train PPO or DDPG on the O-RAN PRB scheduler environment.

Uses Stable-Baselines3. Models are saved to models/ as .zip files.
Episode reward history is saved as a .npy file so the comparison
notebook can plot training curves without re-running training.

Usage
-----
    python train_sb3.py --algo ppo  --scenario uniform
    python train_sb3.py --algo ddpg --scenario heterogeneous
    python train_sb3.py --algo ppo  --scenario bursty --total-steps 100000
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from env.ran_environment import RANEnvironment


# ---------------------------------------------------------------------------
# Callback: records per-episode reward and the global timestep it ended on
# ---------------------------------------------------------------------------

class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.ep_rewards = []
        self.ep_steps   = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.ep_rewards.append(self._ep_reward)
            self.ep_steps.append(self.num_timesteps)
            self._ep_reward = 0.0
        return True


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    total_steps = 50_000,
    model_dir   = "models",
    # shared
    lr          = 3e-4,
    gamma       = 0.99,
    net_arch    = [256, 256],   # match DQN hidden size
    # PPO-specific
    ppo_n_steps     = 512,      # steps per rollout per env
    ppo_batch_size  = 64,
    ppo_n_epochs    = 10,
    ppo_clip_range  = 0.2,
    # DDPG-specific
    ddpg_buffer     = 50_000,
    ddpg_batch_size = 256,
    ddpg_noise_std  = 0.1,      # fraction of action-space range
    ddpg_learn_start= 500,
)


def make_env(scenario):
    return RANEnvironment(scenario=scenario)


def train_ppo(scenario, total_steps, cfg):
    env = make_env(scenario)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate   = cfg["lr"],
        gamma           = cfg["gamma"],
        n_steps         = cfg["ppo_n_steps"],
        batch_size      = cfg["ppo_batch_size"],
        n_epochs        = cfg["ppo_n_epochs"],
        clip_range      = cfg["ppo_clip_range"],
        policy_kwargs   = dict(net_arch=cfg["net_arch"]),
        verbose         = 0,
    )

    callback = EpisodeRewardCallback()
    print(f"Training PPO | scenario={scenario} | total_steps={total_steps}")
    print("-" * 60)
    model.learn(total_timesteps=total_steps, callback=callback)
    return model, callback


def train_ddpg(scenario, total_steps, cfg):
    env = make_env(scenario)

    # Exploration noise: Normal noise scaled to action-space range
    action_dim  = env.action_space.shape[0]
    action_range = float(env.action_space.high[0] - env.action_space.low[0])
    noise_std   = cfg["ddpg_noise_std"] * action_range
    action_noise = NormalActionNoise(
        mean   = np.zeros(action_dim),
        sigma  = np.full(action_dim, noise_std),
    )

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate   = cfg["lr"],
        gamma           = cfg["gamma"],
        buffer_size     = cfg["ddpg_buffer"],
        batch_size      = cfg["ddpg_batch_size"],
        learning_starts = cfg["ddpg_learn_start"],
        action_noise    = action_noise,
        policy_kwargs   = dict(net_arch=cfg["net_arch"]),
        verbose         = 0,
    )

    callback = EpisodeRewardCallback()
    print(f"Training DDPG | scenario={scenario} | total_steps={total_steps}")
    print("-" * 60)
    model.learn(total_timesteps=total_steps, callback=callback)
    return model, callback


def run(algo, scenario, total_steps, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    if algo == "ppo":
        model, cb = train_ppo(scenario, total_steps, DEFAULTS)
    elif algo == "ddpg":
        model, cb = train_ddpg(scenario, total_steps, DEFAULTS)
    else:
        raise ValueError(f"Unknown algo '{algo}'. Choose 'ppo' or 'ddpg'.")

    # save model (.zip) and reward history (.npy)
    model_path  = os.path.join(model_dir, f"{algo}_trained_{scenario}")
    reward_path = os.path.join(model_dir, f"{algo}_rewards_{scenario}.npy")

    model.save(model_path)
    np.save(reward_path, np.array([cb.ep_steps, cb.ep_rewards], dtype=object))

    print(f"Model saved  : {model_path}.zip")
    print(f"Rewards saved: {reward_path}")
    print(f"Episodes     : {len(cb.ep_rewards)}")
    if cb.ep_rewards:
        print(f"Final 10-ep avg reward: {np.mean(cb.ep_rewards[-10:]):.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo",        required=True, choices=["ppo", "ddpg"])
    p.add_argument("--scenario",    required=True,
                   choices=["uniform", "heterogeneous", "bursty"])
    p.add_argument("--total-steps", type=int, default=DEFAULTS["total_steps"])
    p.add_argument("--model-dir",   default=DEFAULTS["model_dir"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.algo, args.scenario,
        args.total_steps, args.model_dir)
