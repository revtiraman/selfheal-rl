"""Episode replay system — generates frame-by-frame HTML for Gradio."""

from __future__ import annotations

from typing import Dict, List, Optional

from ui.visualizer import render_action_log, render_mesh


def generate_replay_frames(
    episode_history: list,
    total_reward: float = 0.0,
) -> List[dict]:
    """Convert episode history into replay frames.

    Each frame contains:
        - mesh_html: HTML for the service mesh at that step
        - log_html: HTML for the action log up to that step
        - step: step number
        - health: system health percentage
    """
    frames = []
    cumulative_reward = 0.0

    for i, record in enumerate(episode_history):
        cumulative_reward += record.reward

        action_info = {
            "target": record.target_service,
            "action_type": record.action_type,
            "success": record.action_success,
        }

        mesh_html = render_mesh(
            service_statuses=record.system_state,
            action_info=action_info,
            step=record.step,
            total_reward=cumulative_reward,
        )

        log_html = render_action_log(episode_history[: i + 1])

        frames.append({
            "mesh_html": mesh_html,
            "log_html": log_html,
            "step": record.step,
            "health": record.system_health,
            "reward": cumulative_reward,
        })

    return frames


def generate_comparison_frames(
    trained_history: list,
    random_history: list,
) -> List[dict]:
    """Generate synced frames for trained vs random comparison."""
    max_steps = max(len(trained_history), len(random_history))
    frames = []

    t_reward = 0.0
    r_reward = 0.0

    for i in range(max_steps):
        # Trained agent frame
        if i < len(trained_history):
            t_rec = trained_history[i]
            t_reward += t_rec.reward
            t_html = render_mesh(
                t_rec.system_state,
                action_info={
                    "target": t_rec.target_service,
                    "action_type": t_rec.action_type,
                    "success": t_rec.action_success,
                },
                step=t_rec.step,
                total_reward=t_reward,
            )
        else:
            t_html = render_mesh(
                trained_history[-1].system_state if trained_history else {},
                step=i,
                total_reward=t_reward,
            )

        # Random agent frame
        if i < len(random_history):
            r_rec = random_history[i]
            r_reward += r_rec.reward
            r_html = render_mesh(
                r_rec.system_state,
                action_info={
                    "target": r_rec.target_service,
                    "action_type": r_rec.action_type,
                    "success": r_rec.action_success,
                },
                step=r_rec.step,
                total_reward=r_reward,
            )
        else:
            r_html = render_mesh(
                random_history[-1].system_state if random_history else {},
                step=i,
                total_reward=r_reward,
            )

        frames.append({
            "trained_html": t_html,
            "random_html": r_html,
            "step": i,
            "trained_reward": t_reward,
            "random_reward": r_reward,
        })

    return frames
