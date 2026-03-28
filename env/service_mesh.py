"""Service mesh simulation — services, dependency graph, and action execution."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    ACTION_SUCCESS_RATES,
    FAILURE_PROGRESSION_STEPS,
    METRIC_NOISE_STD,
    SERVICES,
    SERVICE_NAMES,
    STATUS_DEGRADED,
    STATUS_DOWN,
    STATUS_HEALTHY,
)


@dataclass
class Service:
    """A single microservice with health metrics."""

    name: str
    base_cpu: float
    base_memory: float
    base_latency: float
    depends_on: List[str]
    max_instances: int
    recovery_time: int

    # Dynamic state
    status: float = STATUS_HEALTHY
    cpu: float = 0.0
    memory: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    failure_type: Optional[str] = None
    failure_step: int = 0  # how many steps since failure started
    recovering: bool = False
    recovery_steps_left: int = 0
    instances: int = 1

    def __post_init__(self) -> None:
        self.cpu = self.base_cpu
        self.memory = self.base_memory
        self.latency = self.base_latency
        self.error_rate = 0.0

    def reset(self) -> None:
        """Restore service to healthy baseline."""
        self.status = STATUS_HEALTHY
        self.cpu = self.base_cpu
        self.memory = self.base_memory
        self.latency = self.base_latency
        self.error_rate = 0.0
        self.failure_type = None
        self.failure_step = 0
        self.recovering = False
        self.recovery_steps_left = 0
        self.instances = 1

    @property
    def is_healthy(self) -> bool:
        return self.status >= STATUS_HEALTHY - 0.01

    @property
    def is_degraded(self) -> bool:
        return STATUS_DOWN < self.status < STATUS_HEALTHY - 0.01

    @property
    def is_down(self) -> bool:
        return self.status <= STATUS_DOWN + 0.01

    def noisy_cpu(self) -> float:
        return float(np.clip(self.cpu + np.random.normal(0, METRIC_NOISE_STD), 0, 1))

    def noisy_memory(self) -> float:
        return float(np.clip(self.memory + np.random.normal(0, METRIC_NOISE_STD), 0, 1))

    def noisy_latency(self) -> float:
        noise = np.random.normal(0, METRIC_NOISE_STD * self.base_latency)
        return max(0.0, self.latency + noise)

    def noisy_error_rate(self) -> float:
        return float(np.clip(self.error_rate + np.random.normal(0, METRIC_NOISE_STD), 0, 1))


class ServiceMesh:
    """Simulates a mesh of interdependent microservices."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()
        self.services: Dict[str, Service] = {}
        self._build_services()
        # Pre-compute dependency mappings
        self.dependents: Dict[str, List[str]] = self._build_dependents_map()

    # ── Construction ──────────────────────────────────────────

    def _build_services(self) -> None:
        for name, cfg in SERVICES.items():
            self.services[name] = Service(
                name=name,
                base_cpu=cfg["base_cpu"],
                base_memory=cfg["base_memory"],
                base_latency=cfg["base_latency"],
                depends_on=list(cfg["depends_on"]),
                max_instances=cfg["max_instances"],
                recovery_time=cfg["recovery_time"],
            )

    def _build_dependents_map(self) -> Dict[str, List[str]]:
        """Map each service to the list of services that depend on it."""
        deps: Dict[str, List[str]] = {name: [] for name in self.services}
        for name, svc in self.services.items():
            for dep in svc.depends_on:
                deps[dep].append(name)
        return deps

    # ── Queries ───────────────────────────────────────────────

    def get_service_status(self, name: str) -> dict:
        svc = self.services[name]
        return {
            "name": svc.name,
            "status": svc.status,
            "cpu": svc.cpu,
            "memory": svc.memory,
            "latency": svc.latency,
            "error_rate": svc.error_rate,
            "failure_type": svc.failure_type,
            "recovering": svc.recovering,
            "instances": svc.instances,
        }

    def get_all_statuses(self) -> Dict[str, dict]:
        return {name: self.get_service_status(name) for name in self.services}

    def is_fully_recovered(self) -> bool:
        return all(svc.is_healthy for svc in self.services.values())

    def all_down(self) -> bool:
        return all(svc.is_down for svc in self.services.values())

    def system_health(self) -> float:
        """Overall system health as fraction [0, 1]."""
        return sum(svc.status for svc in self.services.values()) / len(self.services)

    def get_dependency_order(self) -> List[str]:
        """Topological sort — root services first, leaf services last."""
        in_degree = {name: 0 for name in self.services}
        for name, svc in self.services.items():
            in_degree[name] = len(svc.depends_on)

        queue = deque(n for n, d in in_degree.items() if d == 0)
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for dependent in self.dependents[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        return order

    def check_cascade_risk(self, service_name: str) -> List[str]:
        """Return services at risk of cascade if *service_name* degrades/fails."""
        at_risk: List[str] = []
        visited = set()
        queue = deque([service_name])
        while queue:
            current = queue.popleft()
            for dep in self.dependents.get(current, []):
                if dep not in visited:
                    visited.add(dep)
                    at_risk.append(dep)
                    queue.append(dep)
        return at_risk

    def get_down_services(self) -> List[str]:
        return [n for n, s in self.services.items() if s.is_down]

    def get_degraded_services(self) -> List[str]:
        return [n for n, s in self.services.items() if s.is_degraded]

    def get_healthy_services(self) -> List[str]:
        return [n for n, s in self.services.items() if s.is_healthy]

    # ── Failure injection ─────────────────────────────────────

    def inject_failure(self, service_name: str, failure_type: str) -> None:
        """Start a failure on a service (begins at degraded, progresses to down)."""
        svc = self.services[service_name]
        svc.failure_type = failure_type
        svc.failure_step = 0
        svc.status = STATUS_DEGRADED
        svc.error_rate = 0.3
        svc.latency = svc.base_latency * 3
        svc.cpu = min(1.0, svc.base_cpu * 1.5)
        svc.memory = min(1.0, svc.base_memory * 1.3)

    # ── Simulation tick ───────────────────────────────────────

    def tick(self) -> Dict[str, str]:
        """Advance simulation by one step. Returns dict of events that happened."""
        events: Dict[str, str] = {}

        for name, svc in self.services.items():
            if svc.recovering:
                self._tick_recovering(svc, events)
            elif svc.failure_type is not None and not svc.is_healthy:
                self._tick_failing(svc, events)
            elif svc.is_healthy:
                self._tick_healthy(svc)

        # Cascade: down services degrade their dependents
        self._propagate_cascade(events)

        return events

    def _tick_healthy(self, svc: Service) -> None:
        """Small random fluctuations on a healthy service."""
        svc.cpu = float(np.clip(svc.base_cpu + self.rng.normal(0, 0.02), 0, 1))
        svc.memory = float(np.clip(svc.base_memory + self.rng.normal(0, 0.01), 0, 1))
        svc.latency = max(1.0, svc.base_latency + self.rng.normal(0, 2))
        svc.error_rate = float(np.clip(self.rng.normal(0.01, 0.005), 0, 0.05))

    def _tick_failing(self, svc: Service, events: Dict[str, str]) -> None:
        """Failure progresses — service gets worse each step."""
        svc.failure_step += 1
        progression = FAILURE_PROGRESSION_STEPS.get(svc.failure_type, 3)

        progress_frac = min(svc.failure_step / progression, 1.0)

        if progress_frac >= 1.0 and svc.status > STATUS_DOWN:
            # Service crashes
            svc.status = STATUS_DOWN
            svc.error_rate = 1.0
            svc.latency = 9999.0
            svc.cpu = 0.0 if svc.failure_type == "network_partition" else 1.0
            events[svc.name] = "crashed"
        else:
            # Gradual degradation
            svc.status = max(STATUS_DOWN, STATUS_HEALTHY - progress_frac * STATUS_HEALTHY)
            svc.error_rate = min(1.0, 0.1 + progress_frac * 0.9)
            svc.latency = svc.base_latency * (1 + progress_frac * 20)
            svc.cpu = min(1.0, svc.base_cpu + progress_frac * 0.5)
            svc.memory = min(1.0, svc.base_memory + progress_frac * 0.4)
            if svc.status <= STATUS_DEGRADED and svc.status > STATUS_DOWN:
                events[svc.name] = "degrading"

    def _tick_recovering(self, svc: Service, events: Dict[str, str]) -> None:
        """Service is recovering after a successful action."""
        svc.recovery_steps_left -= 1
        if svc.recovery_steps_left <= 0:
            svc.status = STATUS_HEALTHY
            svc.failure_type = None
            svc.failure_step = 0
            svc.recovering = False
            svc.cpu = svc.base_cpu
            svc.memory = svc.base_memory
            svc.latency = svc.base_latency
            svc.error_rate = 0.0
            events[svc.name] = "recovered"
        else:
            # Partial recovery
            frac = 1 - (svc.recovery_steps_left / svc.recovery_time)
            svc.status = STATUS_DEGRADED + frac * (STATUS_HEALTHY - STATUS_DEGRADED)
            svc.error_rate = max(0, 0.3 * (1 - frac))
            svc.latency = svc.base_latency * (1 + 5 * (1 - frac))
            events[svc.name] = "recovering"

    def _propagate_cascade(self, events: Dict[str, str]) -> None:
        """Down services cause their dependents to degrade/crash."""
        for name, svc in self.services.items():
            if svc.recovering or svc.is_down:
                continue
            # Check if any dependency is down
            any_dep_down = any(
                self.services[dep].is_down for dep in svc.depends_on
            )
            any_dep_degraded = any(
                self.services[dep].is_degraded for dep in svc.depends_on
            )
            if any_dep_down and svc.is_healthy:
                # Healthy → degraded (cascade step 1)
                svc.status = STATUS_DEGRADED
                svc.error_rate = 0.4
                svc.latency = svc.base_latency * 5
                if svc.failure_type is None:
                    svc.failure_type = "connection_timeout"
                    svc.failure_step = 0
                events[svc.name] = "cascade_degraded"
            elif any_dep_down and svc.is_degraded:
                # Degraded → down (cascade step 2)
                svc.status = STATUS_DOWN
                svc.error_rate = 1.0
                svc.latency = 9999.0
                events[svc.name] = "cascade_down"

    # ── Action execution ──────────────────────────────────────

    def apply_action(
        self, action_type: str, target_service: str
    ) -> Tuple[bool, str]:
        """Apply an action to a service. Returns (success, description)."""
        svc = self.services[target_service]

        if action_type == "observe":
            return True, f"Observed {target_service}"

        if action_type == "do_nothing":
            return True, "No action taken"

        if svc.is_healthy and action_type != "observe":
            return False, f"{target_service} is already healthy — wasted action"

        if svc.recovering:
            return False, f"{target_service} is already recovering — wasted action"

        # Check if dependencies are still down (wrong order)
        deps_down = [d for d in svc.depends_on if self.services[d].is_down]
        wrong_order = len(deps_down) > 0 and action_type != "observe"

        # Determine success probability
        failure = svc.failure_type or "connection_timeout"
        success_rate = ACTION_SUCCESS_RATES.get((action_type, failure), 0.5)
        success = self.rng.random() < success_rate

        if success:
            svc.recovering = True
            svc.recovery_steps_left = svc.recovery_time
            svc.status = STATUS_DEGRADED  # starts recovering
            svc.error_rate = 0.3
            desc = f"{action_type}({target_service}) succeeded — recovering in {svc.recovery_time} steps"
            if wrong_order:
                desc += f" [WARNING: dependencies still down: {deps_down}]"
        else:
            desc = f"{action_type}({target_service}) failed — {failure} not fixed by {action_type}"

        return success, desc

    # ── Reset ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore all services to healthy state."""
        for svc in self.services.values():
            svc.reset()


if __name__ == "__main__":
    mesh = ServiceMesh()
    print("Services:", list(mesh.services.keys()))
    print("Dependency order:", mesh.get_dependency_order())
    print("All healthy:", mesh.is_fully_recovered())

    mesh.inject_failure("user-db", "disk_full")
    print("\nInjected disk_full on user-db. Watching cascade:")
    for i in range(8):
        events = mesh.tick()
        down = mesh.get_down_services()
        degraded = mesh.get_degraded_services()
        print(f"  Step {i}: DOWN={down}, DEGRADED={degraded}, events={events}")

    print("Cascade happened:", not mesh.is_fully_recovered())
    print("System health:", f"{mesh.system_health():.1%}")
