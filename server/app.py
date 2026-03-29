"""SelfHealRL — FastAPI server (OpenEnv HTTP interface).

Endpoints:
  POST /reset          — start a new episode
  POST /step           — take one action
  GET  /state          — current episode state
  GET  /health         — liveness check
  GET  /tasks          — list available tasks
  POST /reset/{task_id} — start episode for a specific task
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import ACTION_TYPES, MAX_STEPS_PER_EPISODE, SERVICE_NAMES
from core.graders import Grader
from core.tasks import TASKS, TaskGrader, list_tasks
from env.failure_engine import FailureEngine
from env.selfheal_env import SelfHealEnv
from models import (
    ResetResponse,
    SelfHealAction,
    SelfHealObservation,
    SelfHealReward,
    ServiceStatus,
    StateResponse,
    StepResponse,
)

# ─────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SelfHealRL",
    description=(
        "OpenEnv-compatible RL environment for autonomous microservice recovery. "
        "An agent must diagnose and fix cascading failures in a 10-service mesh."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# Session store (single-session for HF Spaces)
# ─────────────────────────────────────────────────────────────────

_session: Dict[str, Any] = {
    "env": None,
    "episode_id": None,
    "task_id": None,
    "done": False,
}

_failure_engine = FailureEngine()


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _build_observation(env: SelfHealEnv, task_id: Optional[str] = None) -> SelfHealObservation:
    """Convert env state → SelfHealObservation."""
    obs_vec = env._get_observation().tolist()
    statuses = env.mesh.get_all_statuses()
    alerts = env.obs_encoder.get_alerts(env.mesh)

    services = []
    for name in SERVICE_NAMES:
        d = statuses[name]
        svc = env.mesh.services[name]
        services.append(ServiceStatus(
            name=name,
            status=d["status"],
            cpu=d.get("cpu", -1.0),
            memory=d.get("memory", -1.0),
            latency_ms=d.get("latency", -1.0),
            error_rate=d.get("error_rate", -1.0),
            observed=name in env.obs_encoder.observed_services or not env.partial_observability,
            alert=name in alerts,
            recovering=svc.recovering,
        ))

    return SelfHealObservation(
        obs_vector=obs_vec,
        services=services,
        step=env.current_step,
        max_steps=MAX_STEPS_PER_EPISODE,
        actions_remaining=env.actions_remaining,
        system_health=env.mesh.system_health(),
        down_services=env.mesh.get_down_services(),
        degraded_services=env.mesh.get_degraded_services(),
        alerts=alerts,
        task_id=task_id,
        difficulty=env.difficulty,
    )


def _build_reward(reward_value: float, env: SelfHealEnv) -> SelfHealReward:
    """Build a SelfHealReward from the scalar step reward."""
    # The env returns a scalar; we return it as total with time_penalty extracted
    return SelfHealReward(
        total=round(reward_value, 4),
        time_penalty=-1.0,
    )


def _get_env() -> SelfHealEnv:
    if _session["env"] is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    return _session["env"]


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — required by HF Spaces ping test."""
    return {"status": "ok", "environment": "selfheal-rl", "version": "1.0.0"}


@app.get("/")
def root():
    """Root redirect to docs."""
    return {
        "name": "SelfHealRL",
        "description": "OpenEnv RL environment for autonomous microservice recovery",
        "docs": "/docs",
        "tasks": "/tasks",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
        },
    }


@app.get("/tasks")
def get_tasks():
    """List all available tasks."""
    return {"tasks": list_tasks()}


@app.post("/reset", response_model=ResetResponse)
def reset(
    difficulty: str = Query(default="EASY", description="EASY | MEDIUM | HARD | CHAOS"),
    partial_observability: bool = Query(default=False),
    seed: Optional[int] = Query(default=None),
):
    """Start a new episode with a random scenario."""
    if difficulty not in ("EASY", "MEDIUM", "HARD", "CHAOS"):
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {difficulty}")

    env = SelfHealEnv(difficulty=difficulty, partial_observability=partial_observability)
    env.reset(seed=seed)

    _session["env"] = env
    _session["episode_id"] = str(uuid.uuid4())
    _session["task_id"] = None
    _session["done"] = False

    obs = _build_observation(env)
    return ResetResponse(
        observation=obs,
        info={
            "episode_id": _session["episode_id"],
            "difficulty": difficulty,
            "scenario": str(env.scenario),
            "partial_observability": partial_observability,
        },
    )


@app.post("/reset/{task_id}", response_model=ResetResponse)
def reset_task(task_id: str, seed: Optional[int] = Query(default=None)):
    """Start a new episode for a specific named task."""
    if task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}"
        )

    task = TASKS[task_id]
    env = SelfHealEnv(
        difficulty=task.difficulty,
        partial_observability=task.partial_observability,
    )
    env.reset(seed=seed)

    # Apply the fixed scenario for this task
    scenario = _failure_engine.generate_from_template(task.scenario_template)
    env.mesh.reset()
    _failure_engine.apply_scenario(env.mesh, scenario)
    env.cascade_sim.reset()
    for svc, _ in scenario.root_failures:
        env.cascade_sim.record_root_cause(svc)
    env.scenario = scenario
    env._prev_down = set(env.mesh.get_down_services())
    env._prev_degraded = set(env.mesh.get_degraded_services())

    _session["env"] = env
    _session["episode_id"] = str(uuid.uuid4())
    _session["task_id"] = task_id
    _session["done"] = False

    obs = _build_observation(env, task_id=task_id)
    return ResetResponse(
        observation=obs,
        info={
            "episode_id": _session["episode_id"],
            "task_id": task_id,
            "task_name": task.name,
            "difficulty": task.difficulty,
            "scenario": scenario.description,
            "partial_observability": task.partial_observability,
            "passing_score": task.passing_score,
        },
    )


@app.post("/step", response_model=StepResponse)
def step(action: SelfHealAction):
    """Take one action in the current episode."""
    env = _get_env()

    if _session["done"]:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new one."
        )

    action_int = action.to_int()
    obs_vec, reward, terminated, truncated, info = env.step(action_int)

    done = terminated or truncated
    _session["done"] = done

    obs = _build_observation(env, task_id=_session["task_id"])
    reward_obj = _build_reward(reward, env)

    # Decode what action was taken
    at = ACTION_TYPES[action_int // len(SERVICE_NAMES)]
    tgt = SERVICE_NAMES[action_int % len(SERVICE_NAMES)]

    step_info: Dict[str, Any] = {
        "episode_id": _session["episode_id"],
        "action_taken": f"{at}({tgt})",
        "action_success": env.episode_history[-1].action_success if env.episode_history else None,
        "step": env.current_step,
    }

    # Add grader results when episode ends
    if done:
        summary = env.get_episode_summary()
        grades = Grader.grade_all(summary)
        step_info["grades"] = {
            k: {"score": v["score"], "passed": v["passed"]}
            for k, v in grades.items()
            if isinstance(v, dict) and "score" in v
        }
        step_info["overall_score"] = grades["overall_score"]
        step_info["fully_recovered"] = env.mesh.is_fully_recovered()

        # Task-specific grade if running a task
        if _session["task_id"]:
            task_grade = TaskGrader.grade(_session["task_id"], summary)
            step_info["task_grade"] = task_grade

    return StepResponse(
        observation=obs,
        reward=reward_obj,
        done=done,
        truncated=truncated,
        info=step_info,
    )


@app.get("/state", response_model=StateResponse)
def state():
    """Return current episode state."""
    env = _get_env()
    s = env.state()
    return StateResponse(
        episode_id=_session["episode_id"] or "unknown",
        task_id=_session["task_id"],
        difficulty=s["difficulty"],
        step=s["step"],
        max_steps=s["max_steps"],
        actions_remaining=s["actions_remaining"],
        done=_session["done"],
        total_reward=s["total_reward"],
        system_health=s["system_health"],
        down_services=s["down_services"],
        scenario=s["scenario"],
    )


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
