"""OpenEnv typed Pydantic models for SelfHealRL.

Defines the three core data contracts:
  - SelfHealAction   — what the agent sends to the environment
  - SelfHealObservation — what the environment returns to the agent
  - SelfHealReward   — structured reward breakdown
  - StepResponse     — full response from POST /step
  - ResetResponse    — full response from POST /reset
  - StateResponse    — full response from GET /state
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from config import ACTION_TYPES, NUM_ACTIONS, OBSERVATION_DIM, SERVICE_NAMES


# ─────────────────────────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────────────────────────

class SelfHealAction(BaseModel):
    """Action sent by the agent to the environment.

    Two equivalent representations are accepted:
      1. Integer encoding: action_int in [0, 59]
         action_int = action_type_idx * 10 + service_idx
      2. Named encoding: action_type + target_service
         These are converted to action_int internally.
    """

    action_int: Optional[int] = Field(
        default=None,
        ge=0,
        lt=NUM_ACTIONS,
        description=(
            f"Discrete action integer in [0, {NUM_ACTIONS - 1}]. "
            f"Encoded as action_type_idx * {len(SERVICE_NAMES)} + service_idx."
        ),
    )
    action_type: Optional[str] = Field(
        default=None,
        description=f"Action type name. One of: {ACTION_TYPES}",
    )
    target_service: Optional[str] = Field(
        default=None,
        description=f"Target service name. One of: {SERVICE_NAMES}",
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v):
        if v is not None and v not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES}, got '{v}'")
        return v

    @field_validator("target_service")
    @classmethod
    def validate_target_service(cls, v):
        if v is not None and v not in SERVICE_NAMES:
            raise ValueError(f"target_service must be one of {SERVICE_NAMES}, got '{v}'")
        return v

    def to_int(self) -> int:
        """Resolve to integer action, regardless of which form was provided."""
        if self.action_int is not None:
            return self.action_int
        if self.action_type is not None and self.target_service is not None:
            return (
                ACTION_TYPES.index(self.action_type) * len(SERVICE_NAMES)
                + SERVICE_NAMES.index(self.target_service)
            )
        raise ValueError(
            "Provide either action_int OR both action_type + target_service"
        )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"action_int": 0},
                {"action_type": "restart", "target_service": "api-gateway"},
                {"action_type": "observe", "target_service": "user-db"},
            ]
        }
    }


# ─────────────────────────────────────────────────────────────────
# Observation — per-service metrics
# ─────────────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    """Health snapshot of a single microservice."""

    name: str = Field(..., description="Service name")
    status: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Health status: 1.0=healthy, 0.5=degraded, 0.0=down, -1.0=unobserved"
    )
    cpu: float = Field(
        ..., ge=-1.0, le=1.0,
        description="CPU utilization [0,1]. -1 if not yet observed."
    )
    memory: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Memory utilization [0,1]. -1 if not yet observed."
    )
    latency_ms: float = Field(
        ..., description="Response latency in ms. -1 if not yet observed."
    )
    error_rate: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Error rate [0,1]. -1 if not yet observed."
    )
    observed: bool = Field(
        ..., description="Whether the agent has observed this service this episode."
    )
    alert: bool = Field(
        ..., description="Whether an alert is active (visible even without observing)."
    )
    recovering: bool = Field(
        ..., description="Whether a recovery action is in progress."
    )


class SelfHealObservation(BaseModel):
    """Observation returned by the environment after each step/reset.

    Contains both the raw vector (for RL agents) and structured
    per-service data (for LLM/rule-based agents).
    """

    # Raw vector for RL agents
    obs_vector: List[float] = Field(
        ...,
        min_length=OBSERVATION_DIM,
        max_length=OBSERVATION_DIM,
        description=f"Flat observation vector of shape ({OBSERVATION_DIM},). "
                    "Values in [-1.0, 1.0]."
    )

    # Structured data for LLM / heuristic agents
    services: List[ServiceStatus] = Field(
        ..., description="Per-service health snapshots (10 services)."
    )

    # Global state
    step: int = Field(..., ge=0, description="Current step number.")
    max_steps: int = Field(..., description="Maximum steps per episode.")
    actions_remaining: int = Field(..., ge=0, description="Remaining action budget.")
    system_health: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of services that are healthy."
    )
    down_services: List[str] = Field(
        ..., description="Names of services currently down."
    )
    degraded_services: List[str] = Field(
        ..., description="Names of services currently degraded."
    )
    alerts: List[str] = Field(
        ..., description="Services with active alerts (visible without observing)."
    )

    # Task context
    task_id: Optional[str] = Field(
        default=None, description="Active task ID, if running a named task."
    )
    difficulty: str = Field(..., description="Scenario difficulty: EASY/MEDIUM/HARD/CHAOS")

    model_config = {
        "json_schema_extra": {
            "example": {
                "obs_vector": [1.0, 0.3, 0.4, 0.05, 0.01, 1.0, 0.0] * 10 + [0.0, 1.0, 1.0, 0.0],
                "services": [
                    {
                        "name": "api-gateway",
                        "status": 1.0, "cpu": 0.3, "memory": 0.4,
                        "latency_ms": 50.0, "error_rate": 0.01,
                        "observed": True, "alert": False, "recovering": False
                    }
                ],
                "step": 0, "max_steps": 30, "actions_remaining": 10,
                "system_health": 0.9, "down_services": ["notification-service"],
                "degraded_services": [], "alerts": ["notification-service"],
                "task_id": "task_easy", "difficulty": "EASY"
            }
        }
    }


# ─────────────────────────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────────────────────────

class SelfHealReward(BaseModel):
    """Structured reward breakdown for a single step."""

    total: float = Field(..., description="Total scalar reward for this step.")

    # Component breakdown
    time_penalty: float = Field(default=0.0, description="-1.0 per step.")
    service_recovered: float = Field(
        default=0.0, description="+10 when a service starts recovering (once per service)."
    )
    root_cause_fixed: float = Field(
        default=0.0, description="+15 when a root cause service is fixed."
    )
    observe_bonus: float = Field(
        default=0.0, description="+1 for first observe of a service."
    )
    observe_penalty: float = Field(
        default=0.0, description="-2 for re-observing same service while services down."
    )
    wasted_action: float = Field(
        default=0.0, description="-3 for acting on a healthy service."
    )
    wrong_order: float = Field(
        default=0.0, description="-5 for fixing downstream before upstream."
    )
    cascade_penalty: float = Field(
        default=0.0, description="-5 × N for causing N new failures."
    )
    do_nothing_penalty: float = Field(
        default=0.0, description="-3 × num_down for doing nothing while services are down."
    )
    full_recovery_bonus: float = Field(
        default=0.0, description="+20 end-of-episode bonus for full recovery."
    )
    timeout_penalty: float = Field(
        default=0.0, description="-10 if episode ends without recovery."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 24.0,
                "time_penalty": -1.0,
                "service_recovered": 10.0,
                "root_cause_fixed": 15.0,
                "observe_bonus": 0.0,
                "observe_penalty": 0.0,
                "wasted_action": 0.0,
                "wrong_order": 0.0,
                "cascade_penalty": 0.0,
                "do_nothing_penalty": 0.0,
                "full_recovery_bonus": 0.0,
                "timeout_penalty": 0.0
            }
        }
    }


# ─────────────────────────────────────────────────────────────────
# HTTP Response models
# ─────────────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    """Response from POST /step."""

    observation: SelfHealObservation
    reward: SelfHealReward
    done: bool = Field(..., description="True if episode has ended.")
    truncated: bool = Field(
        ..., description="True if episode ended due to max_steps (not natural termination)."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional info: action taken, scenario, grader results if done."
    )


class ResetResponse(BaseModel):
    """Response from POST /reset."""

    observation: SelfHealObservation
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Episode info: task_id, scenario description, difficulty."
    )


class StateResponse(BaseModel):
    """Response from GET /state."""

    episode_id: str = Field(..., description="Unique episode identifier.")
    task_id: Optional[str] = Field(default=None, description="Active task ID.")
    difficulty: str = Field(..., description="Current difficulty level.")
    step: int = Field(..., description="Current step number.")
    max_steps: int = Field(..., description="Maximum steps per episode.")
    actions_remaining: int = Field(..., description="Remaining action budget.")
    done: bool = Field(..., description="Whether the episode has ended.")
    total_reward: float = Field(..., description="Cumulative reward so far.")
    system_health: float = Field(..., description="Current system health [0,1].")
    down_services: List[str] = Field(..., description="Currently down services.")
    scenario: Optional[str] = Field(
        default=None, description="Human-readable scenario description."
    )
