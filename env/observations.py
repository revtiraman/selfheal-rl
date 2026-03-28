"""Observation encoding — converts mesh state to flat numpy arrays for the RL agent."""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from config import (
    MAX_STEPS_PER_EPISODE,
    ACTION_BUDGET,
    METRIC_NOISE_STD,
    NUM_SERVICES,
    OBS_GLOBAL,
    OBS_PER_SERVICE,
    OBSERVATION_DIM,
    SERVICE_NAMES,
    STATUS_DEGRADED,
    STATUS_DOWN,
)
from env.service_mesh import ServiceMesh


class ObservationEncoder:
    """Encodes the service mesh state into a flat observation vector."""

    def __init__(self, partial_observability: bool = True) -> None:
        self.partial_observability = partial_observability
        self.observed_services: Set[str] = set()

    def reset(self) -> None:
        self.observed_services.clear()

    def mark_observed(self, service_name: str) -> None:
        self.observed_services.add(service_name)

    def encode(
        self,
        mesh: ServiceMesh,
        current_step: int,
        actions_remaining: int,
        alerts: List[str],
    ) -> np.ndarray:
        """Encode current state into a flat array of shape (OBSERVATION_DIM,).

        Per-service features (7 each):
            0: observed (1.0 if observed or full-obs mode, else 0.0)
            1: cpu (noisy if observed, -1 if not)
            2: memory (noisy if observed, -1 if not)
            3: latency (normalized 0-1, noisy if observed, -1 if not)
            4: error_rate (noisy if observed, -1 if not)
            5: inferred_status (from visible metrics / alerts)
            6: alert_active (1.0 if alert present for this service)

        Global features (4):
            0: time_step (normalized 0-1)
            1: actions_remaining (normalized 0-1)
            2: system_health (0-1)
            3: active_alerts_count (normalized)
        """
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

        # Encode each service
        for i, name in enumerate(SERVICE_NAMES):
            offset = i * OBS_PER_SERVICE
            svc = mesh.services[name]
            has_alert = name in alerts
            can_see = not self.partial_observability or name in self.observed_services

            # Feature 0: observed flag
            obs[offset + 0] = 1.0 if can_see else 0.0

            if can_see:
                # Features 1-4: noisy metrics
                obs[offset + 1] = svc.noisy_cpu()
                obs[offset + 2] = svc.noisy_memory()
                obs[offset + 3] = min(1.0, svc.noisy_latency() / 10000.0)  # normalize
                obs[offset + 4] = svc.noisy_error_rate()
                # Feature 5: actual status (with slight noise)
                obs[offset + 5] = float(np.clip(
                    svc.status + np.random.normal(0, 0.02), 0, 1
                ))
            else:
                # Can't see — fill with -1 (unknown)
                obs[offset + 1] = -1.0
                obs[offset + 2] = -1.0
                obs[offset + 3] = -1.0
                obs[offset + 4] = -1.0
                # Infer status from alerts if available
                if has_alert:
                    obs[offset + 5] = 0.3  # "something is wrong but not sure what"
                else:
                    obs[offset + 5] = -1.0

            # Feature 6: alert active (always visible)
            obs[offset + 6] = 1.0 if has_alert else 0.0

        # Global features
        global_offset = NUM_SERVICES * OBS_PER_SERVICE
        obs[global_offset + 0] = current_step / MAX_STEPS_PER_EPISODE
        obs[global_offset + 1] = actions_remaining / ACTION_BUDGET
        obs[global_offset + 2] = mesh.system_health()
        obs[global_offset + 3] = min(1.0, len(alerts) / NUM_SERVICES)

        return obs

    def get_alerts(self, mesh: ServiceMesh) -> List[str]:
        """Generate alerts based on service metrics.

        Alerts fire for services with high error rates or high latency,
        regardless of whether the agent has observed them.
        """
        alerts: List[str] = []
        for name, svc in mesh.services.items():
            if svc.error_rate > 0.3 or svc.latency > svc.base_latency * 5:
                alerts.append(name)
            elif svc.is_down:
                alerts.append(name)
        return alerts
