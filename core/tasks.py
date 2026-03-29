"""Task definitions for SelfHealRL OpenEnv environment.

Three tasks with increasing difficulty, each with a fixed scenario,
deterministic grader, and a score in [0.0, 1.0].

Task IDs:
  task_easy   — single_fault_recovery
  task_medium — cascade_recovery
  task_hard   — multi_fault_recovery
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from config import MAX_STEPS_PER_EPISODE, SERVICE_NAMES, SERVICES


# ─────────────────────────────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────────────────────────────

@dataclass
class TaskDefinition:
    task_id: str
    name: str
    difficulty: str
    description: str
    scenario_template: str          # which template to use from failure_engine
    partial_observability: bool
    max_steps: int
    action_budget: int
    passing_score: float            # minimum score to "pass" this task
    grader_weights: Dict[str, float]


TASKS: Dict[str, TaskDefinition] = {
    "task_easy": TaskDefinition(
        task_id="task_easy",
        name="Single Fault Recovery",
        difficulty="EASY",
        description=(
            "A single microservice (api-gateway) experiences a sudden CPU spike. "
            "The agent must diagnose and restart it before it causes a full cascade. "
            "Full observability — all service metrics are visible from the start. "
            "Success: restore 100% system health within 10 steps."
        ),
        scenario_template="flash_crash",
        partial_observability=False,
        max_steps=MAX_STEPS_PER_EPISODE,
        action_budget=10,
        passing_score=0.7,
        grader_weights={
            "recovery": 0.40,
            "mttr": 0.25,
            "efficiency": 0.20,
            "diagnosis": 0.15,
        },
    ),

    "task_medium": TaskDefinition(
        task_id="task_medium",
        name="Cascade Recovery",
        difficulty="MEDIUM",
        description=(
            "The user-db fills up (disk_full), triggering a cascade: "
            "auth-service → api-gateway → payment-service all go down. "
            "Partial observability — the agent must observe services to see their metrics. "
            "Success: recover all services in the correct dependency order."
        ),
        scenario_template="database_cascade",
        partial_observability=True,
        max_steps=MAX_STEPS_PER_EPISODE,
        action_budget=10,
        passing_score=0.6,
        grader_weights={
            "recovery": 0.30,
            "cascade_prevention": 0.25,
            "dependency_ordering": 0.25,
            "diagnosis": 0.20,
        },
    ),

    "task_hard": TaskDefinition(
        task_id="task_hard",
        name="Multi-Fault Recovery",
        difficulty="HARD",
        description=(
            "Two root services fail simultaneously: user-db (cpu_spike) and "
            "cache-layer (memory_leak), causing a massive cascade across 8+ services. "
            "Partial observability, tight action budget. "
            "Success: identify both root causes and recover the system efficiently."
        ),
        scenario_template="double_trouble",
        partial_observability=True,
        max_steps=MAX_STEPS_PER_EPISODE,
        action_budget=10,
        passing_score=0.5,
        grader_weights={
            "recovery": 0.25,
            "cascade_prevention": 0.20,
            "dependency_ordering": 0.20,
            "efficiency": 0.15,
            "diagnosis": 0.20,
        },
    ),
}


# ─────────────────────────────────────────────────────────────────
# Task Grader
# ─────────────────────────────────────────────────────────────────

class TaskGrader:
    """Grades a completed episode against a specific task definition."""

    @staticmethod
    def grade(task_id: str, episode_summary: dict) -> dict:
        """
        Grade an episode for the given task.

        Returns:
            {
              "task_id": str,
              "score": float,          # 0.0 – 1.0
              "passed": bool,
              "breakdown": dict,       # per-grader scores
              "details": dict,         # extra info
            }
        """
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(TASKS.keys())}")

        task = TASKS[task_id]
        breakdown = {}

        # ── Recovery score ──────────────────────────────────────
        if "recovery" in task.grader_weights:
            history = episode_summary.get("history", [])
            fully_recovered = episode_summary.get("fully_recovered", False)
            if history:
                last = history[-1]
                recovery_pct = len(last.healthy_services) / len(SERVICE_NAMES)
            else:
                recovery_pct = episode_summary.get("final_health", 0.0)
            breakdown["recovery"] = recovery_pct

        # ── MTTR score ──────────────────────────────────────────
        if "mttr" in task.grader_weights:
            history = episode_summary.get("history", [])
            recovery_steps = [
                i for i, r in enumerate(history)
                if r.action_success and r.action_type not in ("observe", "do_nothing")
            ]
            mean_ttr = (
                sum(recovery_steps) / len(recovery_steps)
                if recovery_steps else MAX_STEPS_PER_EPISODE
            )
            breakdown["mttr"] = max(0.0, 1.0 - mean_ttr / MAX_STEPS_PER_EPISODE)

        # ── Cascade prevention score ────────────────────────────
        if "cascade_prevention" in task.grader_weights:
            history = episode_summary.get("history", [])
            root_causes = set(episode_summary.get("root_causes", []))
            cascades_caused = 0
            prev_down: set = set()
            for record in history:
                curr_down = set(record.down_services)
                for svc in curr_down - prev_down:
                    if svc not in root_causes:
                        cascades_caused += 1
                prev_down = curr_down
            breakdown["cascade_prevention"] = 1.0 - min(
                1.0, cascades_caused / max(1, len(SERVICE_NAMES))
            )

        # ── Dependency ordering score ───────────────────────────
        if "dependency_ordering" in task.grader_weights:
            history = episode_summary.get("history", [])
            correct = incorrect = 0
            for record in history:
                if record.action_type in ("observe", "do_nothing") or not record.action_success:
                    continue
                deps = SERVICES.get(record.target_service, {}).get("depends_on", [])
                if any(d in record.down_services for d in deps):
                    incorrect += 1
                else:
                    correct += 1
            breakdown["dependency_ordering"] = correct / max(1, correct + incorrect)

        # ── Efficiency score ────────────────────────────────────
        if "efficiency" in task.grader_weights:
            history = episode_summary.get("history", [])
            useful = wasted = 0
            for record in history:
                if record.action_type in ("observe", "do_nothing"):
                    continue
                if record.action_success:
                    useful += 1
                else:
                    wasted += 1
            breakdown["efficiency"] = useful / max(1, useful + wasted)

        # ── Diagnosis score ─────────────────────────────────────
        if "diagnosis" in task.grader_weights:
            history = episode_summary.get("history", [])
            root_causes = set(episode_summary.get("root_causes", []))
            identified = any(
                r.action_type not in ("observe", "do_nothing")
                and r.target_service in root_causes
                and r.action_success
                for r in history
            )
            breakdown["diagnosis"] = 1.0 if identified else 0.0

        # ── Weighted total ──────────────────────────────────────
        score = sum(
            breakdown.get(k, 0.0) * w
            for k, w in task.grader_weights.items()
        )
        score = round(min(1.0, max(0.0, score)), 4)

        return {
            "task_id": task_id,
            "task_name": task.name,
            "difficulty": task.difficulty,
            "score": score,
            "passed": score >= task.passing_score,
            "passing_score": task.passing_score,
            "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "details": {
                "fully_recovered": episode_summary.get("fully_recovered", False),
                "final_health": episode_summary.get("final_health", 0.0),
                "steps_taken": episode_summary.get("steps", 0),
                "total_reward": episode_summary.get("total_reward", 0.0),
            },
        }

    @classmethod
    def grade_all_tasks(cls, results: Dict[str, dict]) -> dict:
        """
        Aggregate grades across all tasks.

        Args:
            results: {task_id: episode_summary, ...}

        Returns:
            {
              "tasks": {task_id: grade_result, ...},
              "overall_score": float,   # mean across all tasks
              "all_passed": bool,
            }
        """
        grades = {tid: cls.grade(tid, summary) for tid, summary in results.items()}
        overall = sum(g["score"] for g in grades.values()) / max(1, len(grades))
        return {
            "tasks": grades,
            "overall_score": round(overall, 4),
            "all_passed": all(g["passed"] for g in grades.values()),
        }


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[dict]:
    """Return a list of task metadata dicts (for API / openenv.yaml)."""
    return [
        {
            "task_id": t.task_id,
            "name": t.name,
            "difficulty": t.difficulty,
            "description": t.description,
            "partial_observability": t.partial_observability,
            "max_steps": t.max_steps,
            "action_budget": t.action_budget,
            "passing_score": t.passing_score,
        }
        for t in TASKS.values()
    ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from env.selfheal_env import SelfHealEnv
    from env.failure_engine import FailureEngine

    engine = FailureEngine()

    print("=" * 60)
    print("  SelfHealRL — Task Verification")
    print("=" * 60)

    for task_id, task in TASKS.items():
        env = SelfHealEnv(
            difficulty=task.difficulty,
            partial_observability=task.partial_observability,
        )
        scenario = engine.generate_from_template(task.scenario_template)
        engine.apply_scenario(env.mesh, scenario)

        # Reset with the pre-applied scenario
        obs, _ = env.reset(seed=42)

        # Random agent
        for _ in range(task.max_steps):
            action = env.action_space.sample()
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        summary = env.get_episode_summary()
        grade = TaskGrader.grade(task_id, summary)

        print(f"\n{task_id} ({task.difficulty}): {task.name}")
        print(f"  Score:      {grade['score']:.3f}  (pass >= {task.passing_score})")
        print(f"  Passed:     {grade['passed']}")
        print(f"  Breakdown:  {grade['breakdown']}")
        print(f"  Recovered:  {grade['details']['fully_recovered']}")
        print(f"  Health:     {grade['details']['final_health']:.0%}")
