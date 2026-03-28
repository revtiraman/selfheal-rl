"""Multi-objective reward function for SelfHealRL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from config import MAX_STEPS_PER_EPISODE, REWARD_CONFIG, SERVICE_NAMES

if TYPE_CHECKING:
    from env.selfheal_env import StepRecord


class RewardCalculator:
    """Calculates step-level and episode-level rewards with full breakdown."""

    def __init__(self) -> None:
        self.cfg = REWARD_CONFIG

    def calculate_step_reward(
        self,
        action_type: str,
        target_service: str,
        action_success: bool,
        prev_down: set,
        prev_degraded: set,
        curr_down: set,
        root_causes: set,
        service_deps_down: List[str],
        is_recovering: bool,
        is_healthy: bool,
    ) -> Dict[str, float]:
        """Calculate reward breakdown for a single step."""
        breakdown: Dict[str, float] = {}

        breakdown["time_penalty"] = self.cfg["time_penalty"]

        if action_type == "do_nothing":
            return breakdown

        if action_type == "observe":
            breakdown["observe_bonus"] = self.cfg["observe_bonus"]
            return breakdown

        if action_success and is_recovering:
            breakdown["service_recovered"] = self.cfg["service_recovered"]
            if target_service in root_causes:
                breakdown["root_cause_fixed"] = self.cfg["root_cause_fixed"]
            if target_service in prev_degraded and target_service not in prev_down:
                breakdown["preventive_action_bonus"] = self.cfg["preventive_action_bonus"]

        if is_healthy and not is_recovering:
            breakdown["wasted_action"] = self.cfg["wasted_action"]

        if service_deps_down and action_type not in ("observe", "do_nothing"):
            breakdown["wrong_order"] = self.cfg["wrong_order"]

        new_down = curr_down - prev_down
        if new_down:
            breakdown["cascade_caused"] = self.cfg["cascade_caused"] * len(new_down)

        return breakdown

    def calculate_episode_reward(self, episode_summary: dict) -> Dict[str, float]:
        """Calculate end-of-episode bonuses/penalties."""
        breakdown: Dict[str, float] = {}
        steps = episode_summary["steps"]
        fully_recovered = episode_summary["fully_recovered"]
        history: List[StepRecord] = episode_summary["history"]

        if fully_recovered:
            breakdown["full_recovery"] = self.cfg["full_recovery"]
            efficiency = (MAX_STEPS_PER_EPISODE - steps) / MAX_STEPS_PER_EPISODE
            breakdown["efficiency_bonus"] = efficiency * 10.0

        wasted = sum(
            1 for r in history
            if not r.action_success and r.action_type not in ("observe", "do_nothing")
        )
        if wasted == 0 and fully_recovered:
            breakdown["perfect_run"] = 25.0

        if steps >= MAX_STEPS_PER_EPISODE and not fully_recovered:
            breakdown["timeout_penalty"] = -15.0

        if episode_summary["final_health"] < 0.1:
            breakdown["total_failure"] = -30.0

        return breakdown

    def get_reward_breakdown(self, episode_summary: dict) -> dict:
        """Full reward breakdown for an episode."""
        episode_rewards = self.calculate_episode_reward(episode_summary)
        return {
            "step_total": sum(r.reward for r in episode_summary["history"]),
            "episode_bonuses": episode_rewards,
            "episode_bonus_total": sum(episode_rewards.values()),
            "grand_total": (
                sum(r.reward for r in episode_summary["history"])
                + sum(episode_rewards.values())
            ),
        }
