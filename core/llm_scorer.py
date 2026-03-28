"""LLM-based decision quality scoring with heuristic fallback."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional

from config import ACTION_SUCCESS_RATES, SERVICES

if TYPE_CHECKING:
    from env.selfheal_env import StepRecord


class LLMScorer:
    """Evaluates agent decisions using LLM or heuristic fallback.

    Modes:
        "api"       — Uses HuggingFace Inference API
        "heuristic" — Rule-based scoring (no API needed)
        "mock"      — Fast mock scores for development
        "auto"      — Try API first, fall back to heuristic
    """

    def __init__(self, mode: str = "auto") -> None:
        self.mode = mode
        self._api_available: Optional[bool] = None

    def score_decision(self, decision: dict) -> dict:
        """Score a single agent decision."""
        if self.mode == "mock":
            return self._mock_score(decision)
        if self.mode == "heuristic":
            return self._heuristic_score(decision)
        if self.mode == "api":
            return self._api_score(decision)
        # auto
        if self._api_available is None:
            self._api_available = self._check_api()
        if self._api_available:
            try:
                return self._api_score(decision)
            except Exception:
                self._api_available = False
        return self._heuristic_score(decision)

    def score_episode(self, episode_summary: dict) -> dict:
        """Score the 3 most critical decisions in an episode."""
        history: List[StepRecord] = episode_summary["history"]
        root_causes = set(episode_summary.get("root_causes", []))

        critical_steps = self._pick_critical_steps(history, root_causes)
        step_scores = []
        for record in critical_steps:
            decision = self._record_to_decision(record, episode_summary)
            score = self.score_decision(decision)
            step_scores.append({"step": record.step, "action": f"{record.action_type}({record.target_service})", **score})

        avg_keys = ["root_cause_score", "dependency_score", "action_type_score", "timing_score", "efficiency_score", "overall_score"]
        avg_scores = {}
        for key in avg_keys:
            vals = [s[key] for s in step_scores if key in s]
            avg_scores[key] = sum(vals) / len(vals) if vals else 0.0

        return {"critical_decisions": step_scores, "average_scores": avg_scores}

    def score_strategy(self, episode_summary: dict) -> dict:
        """Evaluate the overall episode strategy."""
        history: List[StepRecord] = episode_summary["history"]
        root_causes = set(episode_summary.get("root_causes", []))

        # Diagnostic approach: did agent observe before acting?
        observe_before_fix = 0
        first_fix_step = -1
        for i, r in enumerate(history):
            if r.action_type == "observe":
                observe_before_fix += 1
            elif r.action_type not in ("do_nothing",):
                first_fix_step = i
                break

        diag_score = min(10, observe_before_fix * 3 + 2)

        # Prioritization: did agent fix root cause first among fix actions?
        fix_actions = [r for r in history if r.action_type not in ("observe", "do_nothing")]
        if fix_actions and fix_actions[0].target_service in root_causes:
            priority_score = 9
        elif any(r.target_service in root_causes for r in fix_actions[:3]):
            priority_score = 6
        else:
            priority_score = 3

        # Patience: did agent wait after actions?
        patience_score = 7  # default
        consecutive_actions = 0
        for r in history:
            if r.action_type not in ("observe", "do_nothing"):
                consecutive_actions += 1
                if consecutive_actions > 3:
                    patience_score = max(2, patience_score - 2)
            else:
                consecutive_actions = 0

        overall = (diag_score + priority_score + patience_score) / 3

        return {
            "diagnostic_approach": diag_score,
            "prioritization": priority_score,
            "patience": patience_score,
            "overall_strategy": round(overall, 1),
        }

    # ── Heuristic scorer ──────────────────────────────────────

    def _heuristic_score(self, decision: dict) -> dict:
        action_type, target = decision.get("action_taken", ("do_nothing", ""))
        root_cause = decision.get("known_root_cause", "")
        system_state = decision.get("system_state", {})
        step = decision.get("step", 0)

        # Root cause identification
        root_score = 9 if target == root_cause else 3

        # Dependency awareness
        deps = SERVICES.get(target, {}).get("depends_on", [])
        deps_down = [d for d in deps if system_state.get(d) == "DOWN"]
        dep_score = 2 if deps_down else 8

        # Action appropriateness
        failure_type = decision.get("failure_type", "connection_timeout")
        success_rate = ACTION_SUCCESS_RATES.get((action_type, failure_type), 0.5)
        action_score = int(success_rate * 10)

        # Timing
        timing_score = max(1, 10 - step)

        # Efficiency
        efficiency_score = 7 if target != root_cause and system_state.get(target) == "HEALTHY" else 8

        overall = (root_score + dep_score + action_score + timing_score + efficiency_score) / 5

        return {
            "root_cause_score": root_score,
            "dependency_score": dep_score,
            "action_type_score": action_score,
            "timing_score": timing_score,
            "efficiency_score": efficiency_score,
            "overall_score": round(overall, 1),
            "reasoning": f"{'Correctly targeted root cause' if target == root_cause else 'Did not target root cause'}. "
                         f"{'Dependencies respected' if not deps_down else f'Dependencies still down: {deps_down}'}. "
                         f"Action success rate for {action_type}/{failure_type}: {success_rate:.0%}.",
        }

    # ── Mock scorer ───────────────────────────────────────────

    def _mock_score(self, decision: dict) -> dict:
        import random
        base = random.uniform(4, 8)
        return {
            "root_cause_score": round(base + random.uniform(-1, 2), 1),
            "dependency_score": round(base + random.uniform(-1, 2), 1),
            "action_type_score": round(base + random.uniform(-1, 1), 1),
            "timing_score": round(base + random.uniform(-2, 1), 1),
            "efficiency_score": round(base + random.uniform(-1, 1), 1),
            "overall_score": round(base, 1),
            "reasoning": "Mock score for development.",
        }

    # ── API scorer ────────────────────────────────────────────

    def _check_api(self) -> bool:
        return os.environ.get("HF_TOKEN") is not None

    def _api_score(self, decision: dict) -> dict:
        import requests

        token = os.environ.get("HF_TOKEN", "")
        url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

        prompt = self._build_prompt(decision)
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.3}}

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()[0]["generated_text"]

        try:
            json_start = text.index("{")
            json_end = text.rindex("}") + 1
            return json.loads(text[json_start:json_end])
        except (ValueError, json.JSONDecodeError):
            return self._heuristic_score(decision)

    def _build_prompt(self, decision: dict) -> str:
        state_str = "\n".join(f"  - {k}: {v}" for k, v in decision.get("system_state", {}).items())
        action_type, target = decision.get("action_taken", ("", ""))
        return (
            f"You are evaluating an AI agent's decision in a microservices recovery scenario.\n\n"
            f"SYSTEM STATE AT STEP {decision.get('step', 0)}:\n{state_str}\n\n"
            f"ROOT CAUSE: {decision.get('known_root_cause', 'unknown')}\n"
            f"AGENT'S ACTION: {action_type}({target})\n\n"
            f"Score 0-10 on: root_cause_score, dependency_score, action_type_score, "
            f"timing_score, efficiency_score, overall_score.\n"
            f"Return ONLY JSON."
        )

    # ── Helpers ───────────────────────────────────────────────

    def _pick_critical_steps(self, history: list, root_causes: set) -> list:
        """Pick up to 3 critical steps: first action, first root-cause fix, last action."""
        critical = []
        first_action = next((r for r in history if r.action_type not in ("observe", "do_nothing")), None)
        if first_action:
            critical.append(first_action)

        root_fix = next((r for r in history if r.target_service in root_causes and r.action_success and r.action_type not in ("observe", "do_nothing")), None)
        if root_fix and root_fix != first_action:
            critical.append(root_fix)

        last_action = next((r for r in reversed(history) if r.action_type not in ("observe", "do_nothing")), None)
        if last_action and last_action not in critical:
            critical.append(last_action)

        return critical[:3]

    def _record_to_decision(self, record, episode_summary: dict) -> dict:
        root_causes = episode_summary.get("root_causes", [])
        state = {}
        for svc, data in record.system_state.items():
            if data["status"] >= 0.9:
                state[svc] = "HEALTHY"
            elif data["status"] > 0.1:
                state[svc] = "DEGRADED"
            else:
                state[svc] = "DOWN"

        return {
            "step": record.step,
            "system_state": state,
            "action_taken": (record.action_type, record.target_service),
            "known_root_cause": root_causes[0] if root_causes else "",
            "failure_type": record.system_state.get(record.target_service, {}).get("failure_type", "unknown"),
        }
