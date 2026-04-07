"""SelfHealRL — Baseline inference script (OpenEnv Hackathon).

Connects to the running SelfHealRL environment container via HTTP and runs
an LLM agent against all 3 tasks.

Environment variables:
  API_BASE_URL   — LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     — Model identifier   (default: gpt-4o-mini)
  HF_TOKEN       — HuggingFace / API key (required, no default)
  ENV_URL        — SelfHealRL env URL (default: http://localhost:8000)

Usage:
  export HF_TOKEN=hf_...
  python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────
# Config from env vars
# ─────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000").rstrip("/")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ─────────────────────────────────────────────────────────────────
# Task definitions (mirrored from openenv.yaml)
# ─────────────────────────────────────────────────────────────────

TASKS = [
    {"task_id": "task_easy",   "passing_score": 0.7},
    {"task_id": "task_medium", "passing_score": 0.6},
    {"task_id": "task_hard",   "passing_score": 0.5},
]

MAX_STEPS = 30

# ─────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous SRE (Site Reliability Engineer) agent.
Your job is to diagnose and fix cascading failures in a microservice system.

SERVICE DEPENDENCY GRAPH:
  api-gateway → auth-service → user-db, cache-layer
  payment-service → auth-service, order-db
  order-service → order-db, cache-layer
  search-service → restaurant-db, cache-layer
  notification-service → user-db

RULES:
1. Fix upstream services (dependencies) BEFORE downstream services.
2. OBSERVE a service before restarting/fixing it — you need to know the failure type.
3. Choose the best action for the failure type:
   - memory_leak     → scale_up
   - cpu_spike       → scale_up or restart
   - bad_deployment  → rollback
   - network_partition → reroute
   - disk_full       → reroute or restart
   - connection_timeout → reroute or restart
4. Do NOT repeat the same action on the same healthy service.
5. Do NOT do_nothing when services are down.

AVAILABLE ACTIONS: restart, scale_up, reroute, rollback, observe, do_nothing

AVAILABLE SERVICES:
  api-gateway, auth-service, payment-service, order-service,
  search-service, notification-service, user-db, cache-layer,
  restaurant-db, order-db

Respond with ONLY valid JSON:
{"action_type": "<action>", "target_service": "<service>", "reasoning": "<1 sentence>"}
"""

VALID_ACTIONS   = ["restart", "scale_up", "reroute", "rollback", "observe", "do_nothing"]
VALID_SERVICES  = [
    "api-gateway", "auth-service", "payment-service", "order-service",
    "search-service", "notification-service", "user-db", "cache-layer",
    "restaurant-db", "order-db",
]

# ─────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────

def env_reset(task_id: str, seed: int = 42) -> dict:
    r = requests.post(f"{ENV_URL}/reset/{task_id}", params={"seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action_type: str, target_service: str, session_id: str = "default") -> dict:
    payload = {"action_type": action_type, "target_service": target_service}
    r = requests.post(
        f"{ENV_URL}/step",
        json=payload,
        params={"session_id": session_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_health() -> bool:
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────
# Observation formatter
# ─────────────────────────────────────────────────────────────────

def format_observation(obs: dict, step: int) -> str:
    lines = [
        f"STEP {step}/{MAX_STEPS} | "
        f"Actions remaining: {obs.get('actions_remaining', '?')} | "
        f"System health: {obs.get('system_health', 0):.0%}",
        "",
        "SERVICE STATUS:",
    ]
    for svc in obs.get("services", []):
        status_val = svc.get("status", -1)
        if status_val >= 0.9:
            status_str = "HEALTHY"
        elif status_val >= 0.4:
            status_str = "DEGRADED"
        elif status_val >= 0.0:
            status_str = "DOWN"
        else:
            status_str = "UNKNOWN"

        alert_str = " ⚠️ALERT" if svc.get("alert") else ""
        if svc.get("observed"):
            ft = f" failure={svc['failure_type']}" if svc.get("failure_type") else ""
            detail = (
                f" cpu={svc.get('cpu', 0):.0%} mem={svc.get('memory', 0):.0%}"
                f" err={svc.get('error_rate', 0):.0%}{ft}"
            )
        else:
            detail = " [NOT OBSERVED]"
        lines.append(f"  {svc['name']:25s} {status_str:8s}{alert_str}{detail}")

    down = obs.get("down_services", [])
    degraded = obs.get("degraded_services", [])
    if down:
        lines.append(f"\nDOWN: {', '.join(down)}")
    if degraded:
        lines.append(f"DEGRADED: {', '.join(degraded)}")
    if not down and not degraded:
        lines.append("\n✅ All services healthy!")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────────────────────────

def llm_action(obs_text: str, history: list) -> tuple[str, str, str]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-3:]:
        messages.append({"role": "assistant", "content": json.dumps(h)})
    messages.append({"role": "user", "content": obs_text})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        parsed = json.loads(match.group() if match else raw)
        action_type    = parsed.get("action_type", "observe")
        target_service = parsed.get("target_service", VALID_SERVICES[0])
        reasoning      = parsed.get("reasoning", "")
        if action_type not in VALID_ACTIONS:
            action_type = "observe"
        if target_service not in VALID_SERVICES:
            target_service = VALID_SERVICES[0]
        return action_type, target_service, reasoning
    except Exception as e:
        return "observe", VALID_SERVICES[0], f"fallback:{e}"

# ─────────────────────────────────────────────────────────────────
# Run one task
# ─────────────────────────────────────────────────────────────────

def run_task(task_id: str, passing_score: float, seed: int = 42) -> dict:
    reset_resp = env_reset(task_id, seed=seed)
    obs        = reset_resp["observation"]
    info       = reset_resp.get("info", {})

    print(json.dumps({
        "type":         "START",
        "task_id":      task_id,
        "scenario":     info.get("scenario", ""),
        "difficulty":   info.get("difficulty", ""),
        "model":        MODEL_NAME,
        "seed":         seed,
    }))
    sys.stdout.flush()

    action_history = []
    step           = 0
    done           = False
    score          = 0.0

    while not done and step < MAX_STEPS:
        obs_text = format_observation(obs, step)
        action_type, target_service, reasoning = llm_action(obs_text, action_history)

        step_resp = env_step(action_type, target_service)
        obs       = step_resp["observation"]
        reward    = step_resp["reward"]["total"]
        done      = step_resp["done"]
        step_info = step_resp.get("info", {})
        step     += 1

        action_history.append({
            "action_type": action_type, "target_service": target_service,
            "reasoning": reasoning, "reward": reward,
        })

        print(json.dumps({
            "type":           "STEP",
            "task_id":        task_id,
            "step":           step,
            "action":         f"{action_type}({target_service})",
            "action_type":    action_type,
            "target_service": target_service,
            "reward":         round(reward, 4),
            "system_health":  round(obs.get("system_health", 0), 4),
            "down_services":  obs.get("down_services", []),
            "done":           done,
            "reasoning":      reasoning,
        }))
        sys.stdout.flush()

        if done:
            score = step_info.get("overall_score", 0.0)
            task_grade = step_info.get("task_grade", {})
            if task_grade:
                score = task_grade.get("score", score)

    print(json.dumps({
        "type":             "END",
        "task_id":          task_id,
        "score":            round(score, 4),
        "passed":           score >= passing_score,
        "passing_score":    passing_score,
        "fully_recovered":  obs.get("system_health", 0) >= 1.0,
        "final_health":     round(obs.get("system_health", 0), 4),
        "steps_taken":      step,
    }))
    sys.stdout.flush()

    return {"task_id": task_id, "score": score, "passed": score >= passing_score,
            "passing_score": passing_score}

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    # Wait for env to be ready
    for attempt in range(12):
        if env_health():
            break
        time.sleep(5)
    else:
        print("ERROR: Environment not reachable at", ENV_URL, file=sys.stderr)
        sys.exit(1)

    results     = {}
    total_start = time.time()

    for task in TASKS:
        task_id      = task["task_id"]
        passing_score = task["passing_score"]
        try:
            grade = run_task(task_id, passing_score, seed=42)
            results[task_id] = grade
        except Exception as e:
            print(json.dumps({
                "type": "END", "task_id": task_id,
                "score": 0.0, "passed": False,
                "passing_score": passing_score, "error": str(e),
            }))
            sys.stdout.flush()
            results[task_id] = {
                "task_id": task_id, "score": 0.0, "passed": False,
                "passing_score": passing_score,
            }

    total_elapsed = time.time() - total_start
    overall       = sum(r["score"] for r in results.values()) / len(results)
    all_passed    = all(r["passed"] for r in results.values())

    print(json.dumps({
        "type":                "SUMMARY",
        "model":               MODEL_NAME,
        "overall_score":       round(overall, 4),
        "all_passed":          all_passed,
        "total_time_seconds":  round(total_elapsed, 1),
        "tasks": {
            tid: {"score": r["score"], "passed": r["passed"]}
            for tid, r in results.items()
        },
    }))
    sys.stdout.flush()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
