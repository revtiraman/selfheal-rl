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
# Session store — supports parallel evaluators via session_id
# ─────────────────────────────────────────────────────────────────

def _new_session() -> Dict[str, Any]:
    return {"env": None, "episode_id": None, "task_id": None, "done": False}

_sessions: Dict[str, Dict[str, Any]] = {"default": _new_session()}
_failure_engine = FailureEngine()


def _get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = _new_session()
    return _sessions[session_id]


def _get_active_env(session_id: str) -> SelfHealEnv:
    sess = _get_session(session_id)
    if sess["env"] is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    return sess["env"]


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
        is_observed = name in env.obs_encoder.observed_services or not env.partial_observability
        services.append(ServiceStatus(
            name=name,
            status=d["status"],
            cpu=d.get("cpu", -1.0),
            memory=d.get("memory", -1.0),
            latency_ms=d.get("latency", -1.0),
            error_rate=d.get("error_rate", -1.0),
            observed=is_observed,
            alert=name in alerts,
            recovering=svc.recovering,
            failure_type=svc.failure_type if is_observed else None,
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




# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — OpenEnv spec requires status: 'healthy'."""
    return {"status": "healthy", "environment": "selfheal-rl", "version": "1.0.0"}


@app.get("/")
def root():
    return {
        "name": "SelfHealRL",
        "description": "OpenEnv RL environment for autonomous microservice recovery",
        "docs": "/docs",
        "tasks": "/tasks",
    }


@app.get("/metadata")
def metadata():
    """OpenEnv spec: returns name and description."""
    return {
        "name": "SelfHealRL",
        "description": (
            "An RL environment where an agent must diagnose and fix cascading failures "
            "in a 10-service microservice mesh."
        ),
        "version": "1.0.0",
        "author": "revtiraman",
        "tasks": list_tasks(),
    }


@app.get("/schema")
def schema():
    """OpenEnv spec: returns action, observation, and state schemas."""
    return {
        "action": {
            "type": "object",
            "description": "Action sent by the agent",
            "properties": {
                "action_int": {"type": "integer", "minimum": 0, "maximum": 59},
                "action_type": {"type": "string", "enum": list(ACTION_TYPES)},
                "target_service": {"type": "string", "enum": list(SERVICE_NAMES)},
            },
        },
        "observation": {
            "type": "object",
            "description": "Observation returned by the environment",
            "properties": {
                "obs_vector": {"type": "array", "items": {"type": "number"}, "minItems": 104, "maxItems": 104},
                "services": {"type": "array"},
                "system_health": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "down_services": {"type": "array", "items": {"type": "string"}},
                "step": {"type": "integer"},
                "actions_remaining": {"type": "integer"},
            },
        },
        "state": {
            "type": "object",
            "description": "Current episode state",
            "properties": {
                "episode_id": {"type": "string"},
                "step": {"type": "integer"},
                "done": {"type": "boolean"},
                "system_health": {"type": "number"},
                "total_reward": {"type": "number"},
            },
        },
    }


class MCPRequest(BaseModel):
    jsonrpc: Optional[str] = "2.0"
    id: Optional[Any] = None
    method: Optional[str] = None
    params: Optional[Any] = None

@app.post("/mcp")
def mcp(request: MCPRequest = None):
    """OpenEnv spec: MCP JSON-RPC endpoint."""
    req_id = request.id if request else None
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "name": "SelfHealRL",
            "description": "OpenEnv RL environment for autonomous microservice recovery",
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
    session_id: str = Query(default="default", description="Session ID for parallel evaluators"),
):
    """Start a new episode with a random scenario."""
    if difficulty not in ("EASY", "MEDIUM", "HARD", "CHAOS"):
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {difficulty}")

    env = SelfHealEnv(difficulty=difficulty, partial_observability=partial_observability)
    env.reset(seed=seed)

    sess = _get_session(session_id)
    episode_id = str(uuid.uuid4())
    sess["env"] = env
    sess["episode_id"] = episode_id
    sess["task_id"] = None
    sess["done"] = False

    obs = _build_observation(env)
    return ResetResponse(
        observation=obs,
        info={
            "session_id": session_id,
            "episode_id": episode_id,
            "difficulty": difficulty,
            "scenario": str(env.scenario),
            "partial_observability": partial_observability,
        },
    )


@app.post("/reset/{task_id}", response_model=ResetResponse)
def reset_task(
    task_id: str,
    seed: Optional[int] = Query(default=None),
    session_id: str = Query(default="default", description="Session ID for parallel evaluators"),
):
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

    sess = _get_session(session_id)
    episode_id = str(uuid.uuid4())
    sess["env"] = env
    sess["episode_id"] = episode_id
    sess["task_id"] = task_id
    sess["done"] = False

    obs = _build_observation(env, task_id=task_id)
    return ResetResponse(
        observation=obs,
        info={
            "session_id": session_id,
            "episode_id": episode_id,
            "task_id": task_id,
            "task_name": task.name,
            "difficulty": task.difficulty,
            "scenario": scenario.description,
            "partial_observability": task.partial_observability,
            "passing_score": task.passing_score,
        },
    )


@app.post("/step", response_model=StepResponse)
def step(
    action: SelfHealAction,
    session_id: str = Query(default="default", description="Session ID for parallel evaluators"),
):
    """Take one action in the current episode."""
    env = _get_active_env(session_id)
    sess = _get_session(session_id)

    if sess["done"]:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new one."
        )

    action_int = action.to_int()
    obs_vec, reward, terminated, truncated, info = env.step(action_int)

    done = terminated or truncated
    sess["done"] = done

    obs = _build_observation(env, task_id=sess["task_id"])
    reward_obj = _build_reward(reward, env)

    at = ACTION_TYPES[action_int // len(SERVICE_NAMES)]
    tgt = SERVICE_NAMES[action_int % len(SERVICE_NAMES)]

    step_info: Dict[str, Any] = {
        "session_id": session_id,
        "episode_id": sess["episode_id"],
        "action_taken": f"{at}({tgt})",
        "action_success": env.episode_history[-1].action_success if env.episode_history else None,
        "step": env.current_step,
    }

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

        if sess["task_id"]:
            task_grade = TaskGrader.grade(sess["task_id"], summary)
            step_info["task_grade"] = task_grade

    return StepResponse(
        observation=obs,
        reward=reward_obj,
        done=done,
        truncated=truncated,
        info=step_info,
    )


@app.get("/state", response_model=StateResponse)
def state(
    session_id: str = Query(default="default", description="Session ID for parallel evaluators"),
):
    """Return current episode state."""
    env = _get_active_env(session_id)
    sess = _get_session(session_id)
    s = env.state()
    return StateResponse(
        episode_id=sess["episode_id"] or "unknown",
        task_id=sess["task_id"],
        difficulty=s["difficulty"],
        step=s["step"],
        max_steps=s["max_steps"],
        actions_remaining=s["actions_remaining"],
        done=sess["done"],
        total_reward=s["total_reward"],
        system_health=s["system_health"],
        down_services=s["down_services"],
        scenario=s["scenario"],
    )


# ─────────────────────────────────────────────────────────────────
# Evaluate endpoint — run heuristic agent, return grade directly
# ─────────────────────────────────────────────────────────────────

def _run_heuristic_episode(task_id: str, seed: int) -> dict:
    """Run one episode with the heuristic agent and return the task grade."""
    from core.heuristic_agent import HeuristicAgent

    task = TASKS[task_id]
    engine = FailureEngine()
    env = SelfHealEnv(
        difficulty=task.difficulty,
        partial_observability=task.partial_observability,
    )
    env.reset(seed=seed)

    scenario = engine.generate_from_template(task.scenario_template)
    env.mesh.reset()
    engine.apply_scenario(env.mesh, scenario)
    env.cascade_sim.reset()
    for svc, _ in scenario.root_failures:
        env.cascade_sim.record_root_cause(svc)
    env.scenario = scenario
    env._prev_down = set(env.mesh.get_down_services())
    env._prev_degraded = set(env.mesh.get_degraded_services())

    agent = HeuristicAgent()
    agent.reset()

    for _ in range(task.max_steps):
        statuses = env.mesh.get_all_statuses()
        act_type, target = agent.act(statuses)
        if act_type == "observe":
            svc_data = env.mesh.services.get(target)
            if svc_data:
                agent.record_observation(target, svc_data.failure_type or "unknown")
        action_int = agent.action_to_int(act_type, target)
        _, _, term, trunc, _ = env.step(action_int)
        if term or trunc:
            break

    summary = env.get_episode_summary()
    return TaskGrader.grade(task_id, summary)


@app.post("/evaluate/{task_id}")
def evaluate_task(
    task_id: str,
    num_episodes: int = Query(default=5, ge=1, le=20, description="Episodes to average over"),
    seed: Optional[int] = Query(default=None),
):
    """Run the built-in heuristic agent against a task and return the grade.

    Useful for verifying the environment works correctly without an external agent.
    Returns averaged scores across num_episodes.
    """
    if task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}"
        )

    base_seed = seed if seed is not None else 42
    grades = [_run_heuristic_episode(task_id, seed=base_seed + i) for i in range(num_episodes)]

    task = TASKS[task_id]
    avg_score = sum(g["score"] for g in grades) / len(grades)
    pass_rate = sum(1 for g in grades if g["passed"]) / len(grades)

    # Average breakdown per grader
    all_keys = grades[0]["breakdown"].keys()
    avg_breakdown = {
        k: round(sum(g["breakdown"].get(k, 0.0) for g in grades) / len(grades), 4)
        for k in all_keys
    }

    return {
        "task_id": task_id,
        "task_name": task.name,
        "difficulty": task.difficulty,
        "num_episodes": num_episodes,
        "avg_score": round(avg_score, 4),
        "pass_rate": round(pass_rate, 4),
        "passed": avg_score >= task.passing_score,
        "passing_score": task.passing_score,
        "avg_breakdown": avg_breakdown,
        "agent": "heuristic",
    }


@app.post("/evaluate")
def evaluate_all(
    num_episodes: int = Query(default=5, ge=1, le=20),
    seed: Optional[int] = Query(default=None),
):
    """Run the heuristic agent against all tasks and return overall score."""
    base_seed = seed if seed is not None else 42
    results = {}
    for tid in TASKS:
        grades = [_run_heuristic_episode(tid, seed=base_seed + i) for i in range(num_episodes)]
        task = TASKS[tid]
        avg_score = sum(g["score"] for g in grades) / len(grades)
        results[tid] = {
            "avg_score": round(avg_score, 4),
            "pass_rate": round(sum(1 for g in grades if g["passed"]) / len(grades), 4),
            "passed": avg_score >= task.passing_score,
            "passing_score": task.passing_score,
        }

    overall = sum(r["avg_score"] for r in results.values()) / len(results)
    return {
        "tasks": results,
        "overall_score": round(overall, 4),
        "all_passed": all(r["passed"] for r in results.values()),
        "num_episodes_per_task": num_episodes,
        "agent": "heuristic",
    }


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
