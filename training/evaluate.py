"""Evaluation utilities — run episodes, compare agents, export results."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from config import MAX_STEPS_PER_EPISODE
from core.graders import Grader
from core.metrics import EpisodeMetrics
from env.selfheal_env import SelfHealEnv


def run_episode(
    env: SelfHealEnv,
    model=None,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> dict:
    """Run a single episode. If model is None, uses random actions."""
    obs, info = env.reset(seed=seed)
    metrics = EpisodeMetrics()
    done = False

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
        else:
            action = env.action_space.sample()

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

    summary = env.get_episode_summary()
    for rec in summary["history"]:
        metrics.record_step(rec)
    metrics.finalize(summary)
    grades = Grader.grade_all(summary)

    return {
        "metrics": metrics,
        "grades": grades,
        "summary": summary,
    }


def evaluate_agent(
    model=None,
    num_episodes: int = 100,
    difficulty: str = "MEDIUM",
    partial_observability: bool = True,
) -> dict:
    """Run multiple episodes and return aggregate stats."""
    env = SelfHealEnv(difficulty=difficulty, partial_observability=partial_observability)

    rewards = []
    recovery_times = []
    successes = []
    grade_scores = []
    efficiency_scores = []

    for i in range(num_episodes):
        result = run_episode(env, model=model, seed=i)
        m = result["metrics"]
        g = result["grades"]

        rewards.append(m.total_reward)
        recovery_times.append(m.mean_time_to_recovery)
        successes.append(float(m.fully_recovered))
        grade_scores.append(g["overall_score"])
        efficiency_scores.append(g["efficiency"]["score"])

    return {
        "num_episodes": num_episodes,
        "difficulty": difficulty,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_recovery_time": float(np.mean(recovery_times)),
        "success_rate": float(np.mean(successes)),
        "mean_grade_score": float(np.mean(grade_scores)),
        "mean_efficiency": float(np.mean(efficiency_scores)),
    }


def compare_agents(
    trained_model,
    num_episodes: int = 100,
    difficulty: str = "MEDIUM",
) -> dict:
    """Side-by-side comparison of trained agent vs random baseline."""
    print(f"Evaluating trained agent ({num_episodes} episodes, {difficulty})...")
    trained_stats = evaluate_agent(trained_model, num_episodes, difficulty)

    print(f"Evaluating random baseline ({num_episodes} episodes, {difficulty})...")
    random_stats = evaluate_agent(None, num_episodes, difficulty)

    improvement = {
        "reward_improvement": trained_stats["mean_reward"] - random_stats["mean_reward"],
        "success_rate_improvement": trained_stats["success_rate"] - random_stats["success_rate"],
        "recovery_time_improvement": random_stats["mean_recovery_time"] - trained_stats["mean_recovery_time"],
    }

    return {
        "trained": trained_stats,
        "random": random_stats,
        "improvement": improvement,
    }
