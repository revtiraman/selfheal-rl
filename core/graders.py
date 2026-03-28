"""Programmatic graders — 6 pass/fail checks for episode quality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from config import MAX_STEPS_PER_EPISODE, SERVICE_NAMES, SERVICES

if TYPE_CHECKING:
    from env.selfheal_env import StepRecord


class Grader:
    """Six programmatic graders that evaluate agent performance."""

    @staticmethod
    def grade_recovery(episode_summary: dict) -> dict:
        """Grade 1: Did the agent recover the system? PASS: >= 80%"""
        history: List[StepRecord] = episode_summary["history"]
        fully_recovered = episode_summary["fully_recovered"]

        if history:
            last = history[-1]
            recovery_pct = len(last.healthy_services) / len(SERVICE_NAMES)
        else:
            recovery_pct = episode_summary["final_health"]

        return {
            "grader": "recovery",
            "full_recovery": fully_recovered,
            "recovery_percentage": recovery_pct,
            "passed": recovery_pct >= 0.8,
            "score": recovery_pct,
        }

    @staticmethod
    def grade_mttr(episode_summary: dict) -> dict:
        """Grade 2: Mean Time to Recovery. PASS: <= 5 steps"""
        history: List[StepRecord] = episode_summary["history"]

        recovery_steps: List[int] = []
        for i, record in enumerate(history):
            if record.action_success and record.action_type not in ("observe", "do_nothing"):
                recovery_steps.append(i)

        mean_ttr = sum(recovery_steps) / len(recovery_steps) if recovery_steps else MAX_STEPS_PER_EPISODE
        max_ttr = max(recovery_steps) if recovery_steps else MAX_STEPS_PER_EPISODE

        return {
            "grader": "mttr",
            "mean_time_to_recovery": mean_ttr,
            "max_time_to_recovery": max_ttr,
            "passed": mean_ttr <= 5,
            "score": max(0.0, 1.0 - mean_ttr / MAX_STEPS_PER_EPISODE),
        }

    @staticmethod
    def grade_cascade_prevention(episode_summary: dict) -> dict:
        """Grade 3: Did the agent prevent cascades? PASS: cascades_caused == 0"""
        history: List[StepRecord] = episode_summary["history"]
        root_causes = set(episode_summary.get("root_causes", []))

        cascades_prevented = 0
        cascades_caused = 0
        prev_down: set = set()

        for record in history:
            curr_down = set(record.down_services)
            for svc in curr_down - prev_down:
                if svc not in root_causes:
                    cascades_caused += 1
            if record.action_success and record.target_service in record.degraded_services:
                cascades_prevented += 1
            prev_down = curr_down

        total = cascades_caused + cascades_prevented
        prevention_rate = cascades_prevented / max(1, total)

        return {
            "grader": "cascade_prevention",
            "cascades_prevented": cascades_prevented,
            "cascades_caused": cascades_caused,
            "prevention_rate": prevention_rate,
            "passed": cascades_caused == 0,
            "score": 1.0 - min(1.0, cascades_caused / max(1, len(SERVICE_NAMES))),
        }

    @staticmethod
    def grade_dependency_ordering(episode_summary: dict) -> dict:
        """Grade 4: Fix upstream before downstream? PASS: score >= 0.8"""
        history: List[StepRecord] = episode_summary["history"]

        correct = 0
        incorrect = 0

        for record in history:
            if record.action_type in ("observe", "do_nothing") or not record.action_success:
                continue
            target = record.target_service
            deps = SERVICES.get(target, {}).get("depends_on", [])
            if any(d in record.down_services for d in deps):
                incorrect += 1
            else:
                correct += 1

        total = correct + incorrect
        score = correct / max(1, total)

        return {
            "grader": "dependency_ordering",
            "correct_orders": correct,
            "incorrect_orders": incorrect,
            "ordering_score": score,
            "passed": score >= 0.8,
            "score": score,
        }

    @staticmethod
    def grade_efficiency(episode_summary: dict) -> dict:
        """Grade 5: Action efficiency. PASS: score >= 0.7"""
        history: List[StepRecord] = episode_summary["history"]

        total = 0
        useful = 0
        wasted = 0
        for record in history:
            if record.action_type in ("observe", "do_nothing"):
                continue
            total += 1
            if record.action_success:
                useful += 1
            else:
                wasted += 1

        score = useful / max(1, total)

        return {
            "grader": "efficiency",
            "total_actions": total,
            "useful_actions": useful,
            "wasted_actions": wasted,
            "efficiency_score": score,
            "passed": score >= 0.7,
            "score": score,
        }

    @staticmethod
    def grade_diagnosis(episode_summary: dict) -> dict:
        """Grade 6: Root cause identification. PASS: identified == True"""
        history: List[StepRecord] = episode_summary["history"]
        root_causes = set(episode_summary.get("root_causes", []))

        identified = False
        steps_to_diag = -1
        relevant_obs = 0
        total_obs = 0

        for i, record in enumerate(history):
            if record.action_type == "observe":
                total_obs += 1
                if record.target_service in root_causes:
                    relevant_obs += 1
            if (
                record.action_type not in ("observe", "do_nothing")
                and record.target_service in root_causes
                and record.action_success
            ):
                if not identified:
                    identified = True
                    steps_to_diag = i

        return {
            "grader": "diagnosis",
            "root_cause_identified": identified,
            "steps_to_diagnosis": steps_to_diag,
            "diagnosis_accuracy": relevant_obs / max(1, total_obs),
            "passed": identified,
            "score": 1.0 if identified else 0.0,
        }

    @classmethod
    def grade_all(cls, episode_summary: dict) -> dict:
        """Run all 6 graders and return aggregate results."""
        results = {
            "recovery": cls.grade_recovery(episode_summary),
            "mttr": cls.grade_mttr(episode_summary),
            "cascade_prevention": cls.grade_cascade_prevention(episode_summary),
            "dependency_ordering": cls.grade_dependency_ordering(episode_summary),
            "efficiency": cls.grade_efficiency(episode_summary),
            "diagnosis": cls.grade_diagnosis(episode_summary),
        }

        weights = {
            "recovery": 0.25, "mttr": 0.15, "cascade_prevention": 0.15,
            "dependency_ordering": 0.20, "efficiency": 0.10, "diagnosis": 0.15,
        }
        overall_score = sum(results[k]["score"] * weights[k] for k in results)

        results["overall_pass"] = all(r["passed"] for r in results.values())
        results["overall_score"] = overall_score
        return results
