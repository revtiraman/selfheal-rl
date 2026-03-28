"""Episode metrics tracking and export."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from env.selfheal_env import StepRecord


@dataclass
class EpisodeMetrics:
    """Tracks all metrics for a single episode."""

    episode_id: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    difficulty: str = ""

    total_steps: int = 0
    total_reward: float = 0.0
    fully_recovered: bool = False
    final_health: float = 0.0

    total_actions: int = 0
    useful_actions: int = 0
    wasted_actions: int = 0
    observe_actions: int = 0

    services_recovered: int = 0
    root_cause_fixed: bool = False
    steps_to_root_cause: int = -1
    mean_time_to_recovery: float = 0.0

    step_rewards: List[float] = field(default_factory=list)
    step_health: List[float] = field(default_factory=list)
    step_actions: List[str] = field(default_factory=list)

    def record_step(self, record: StepRecord) -> None:
        self.step_rewards.append(record.reward)
        self.step_health.append(record.system_health)
        self.step_actions.append(f"{record.action_type}({record.target_service})")

        if record.action_type == "observe":
            self.observe_actions += 1
        elif record.action_type != "do_nothing":
            self.total_actions += 1
            if record.action_success:
                self.useful_actions += 1
            else:
                self.wasted_actions += 1

    def finalize(self, episode_summary: dict) -> None:
        self.end_time = time.time()
        self.total_steps = episode_summary["steps"]
        self.total_reward = episode_summary["total_reward"]
        self.fully_recovered = episode_summary["fully_recovered"]
        self.final_health = episode_summary["final_health"]
        self.difficulty = episode_summary["scenario"].difficulty if episode_summary["scenario"] else ""

        history: List[StepRecord] = episode_summary["history"]
        root_causes = set(episode_summary.get("root_causes", []))

        recovery_times: List[int] = []
        for i, record in enumerate(history):
            if record.action_success and record.target_service in root_causes:
                if self.steps_to_root_cause < 0:
                    self.steps_to_root_cause = i
                self.root_cause_fixed = True
            if record.action_success and record.action_type not in ("observe", "do_nothing"):
                self.services_recovered += 1
                recovery_times.append(i)

        if recovery_times:
            self.mean_time_to_recovery = sum(recovery_times) / len(recovery_times)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "difficulty": self.difficulty,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "fully_recovered": self.fully_recovered,
            "final_health": self.final_health,
            "total_actions": self.total_actions,
            "useful_actions": self.useful_actions,
            "wasted_actions": self.wasted_actions,
            "services_recovered": self.services_recovered,
            "root_cause_fixed": self.root_cause_fixed,
            "steps_to_root_cause": self.steps_to_root_cause,
            "mean_time_to_recovery": self.mean_time_to_recovery,
            "step_rewards": self.step_rewards,
            "step_health": self.step_health,
            "step_actions": self.step_actions,
            "duration_seconds": self.end_time - self.start_time,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
