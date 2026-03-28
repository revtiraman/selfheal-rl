"""Cascade simulation — predicts and tracks cascade propagation through the mesh."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from config import SERVICES, SERVICE_NAMES
from env.service_mesh import ServiceMesh


@dataclass
class CascadeEvent:
    """Records a single cascade event."""

    step: int
    source_service: str
    affected_service: str
    new_status: str  # "degraded" or "down"


class CascadeSimulator:
    """Tracks and predicts cascade propagation through the service mesh."""

    def __init__(self) -> None:
        self.cascade_history: List[CascadeEvent] = []
        self.root_causes: Set[str] = set()

    def reset(self) -> None:
        self.cascade_history.clear()
        self.root_causes.clear()

    def record_root_cause(self, service_name: str) -> None:
        """Register a service as a root cause of failure."""
        self.root_causes.add(service_name)

    def cascade_step(self, mesh: ServiceMesh, step: int) -> List[CascadeEvent]:
        """Run one step of cascade propagation and record events.

        This should be called AFTER mesh.tick() to record what cascaded.
        The actual cascade logic lives in ServiceMesh._propagate_cascade();
        this method just observes and records.
        """
        new_events: List[CascadeEvent] = []

        for name, svc in mesh.services.items():
            if name in self.root_causes:
                continue  # root causes are not cascade events

            if svc.is_down or svc.is_degraded:
                # Check if this is caused by a dependency being down
                for dep_name in svc.depends_on:
                    dep = mesh.services[dep_name]
                    if dep.is_down:
                        status = "down" if svc.is_down else "degraded"
                        event = CascadeEvent(
                            step=step,
                            source_service=dep_name,
                            affected_service=name,
                            new_status=status,
                        )
                        # Only record if this is a new event
                        if not self._already_recorded(name, status):
                            new_events.append(event)
                            self.cascade_history.append(event)
                        break  # one source is enough

        return new_events

    def predict_cascade(
        self, mesh: ServiceMesh, failed_service: str
    ) -> Dict[str, int]:
        """Predict which services will be affected and when (steps until affected).

        Uses BFS through the dependency graph, assuming 2 steps per hop
        (1 step to degrade, 1 step to go down, then dependents degrade).
        """
        predictions: Dict[str, int] = {}
        visited: Set[str] = set()
        # (service_name, steps_until_affected)
        queue: deque[Tuple[str, int]] = deque()

        # Start from direct dependents of the failed service
        dependents = self._get_dependents(failed_service)
        for dep in dependents:
            queue.append((dep, 1))  # 1 step to start degrading

        while queue:
            service, steps = queue.popleft()
            if service in visited:
                continue
            visited.add(service)
            predictions[service] = steps

            # This service will go DOWN ~2 steps after degrading
            steps_until_down = steps + 2
            for next_dep in self._get_dependents(service):
                if next_dep not in visited:
                    queue.append((next_dep, steps_until_down))

        return predictions

    def get_cascade_chain(self, mesh: ServiceMesh) -> Dict[str, List[str]]:
        """Return the cascade chain: root_cause → [affected services in order].

        Returns a dict mapping each root cause to its cascade chain.
        """
        chains: Dict[str, List[str]] = {}

        for root in self.root_causes:
            chain: List[str] = []
            visited: Set[str] = set()
            queue: deque[str] = deque([root])

            while queue:
                current = queue.popleft()
                for dep in self._get_dependents(current):
                    if dep not in visited and dep not in self.root_causes:
                        visited.add(dep)
                        chain.append(dep)
                        queue.append(dep)

            chains[root] = chain

        return chains

    def get_affected_services(self) -> Set[str]:
        """Return all services that have been affected by cascades."""
        return {e.affected_service for e in self.cascade_history}

    def was_cascade_caused(self, service: str) -> bool:
        """Check if a service's failure was caused by cascade (not root cause)."""
        return service not in self.root_causes and any(
            e.affected_service == service for e in self.cascade_history
        )

    def get_root_cause_for(self, service: str) -> Optional[str]:
        """Trace back to find the root cause for a cascade-affected service."""
        if service in self.root_causes:
            return service

        for event in self.cascade_history:
            if event.affected_service == service:
                return self.get_root_cause_for(event.source_service)

        return None

    # ── Helpers ───────────────────────────────────────────────

    def _get_dependents(self, service_name: str) -> List[str]:
        """Get services that directly depend on the given service."""
        return [
            name
            for name, cfg in SERVICES.items()
            if service_name in cfg["depends_on"]
        ]

    def _already_recorded(self, service: str, status: str) -> bool:
        """Check if we already recorded this cascade event."""
        return any(
            e.affected_service == service and e.new_status == status
            for e in self.cascade_history
        )


if __name__ == "__main__":
    from env.service_mesh import ServiceMesh

    mesh = ServiceMesh()
    sim = CascadeSimulator()

    # Predict cascade before it happens
    predictions = sim.predict_cascade(mesh, "user-db")
    print("Predicted cascade from user-db failure:")
    for svc, steps in sorted(predictions.items(), key=lambda x: x[1]):
        print(f"  {svc}: affected in ~{steps} steps")

    # Now actually inject and watch
    mesh.inject_failure("user-db", "disk_full")
    sim.record_root_cause("user-db")

    print("\nActual cascade:")
    for step in range(10):
        mesh.tick()
        events = sim.cascade_step(mesh, step)
        if events:
            for e in events:
                print(f"  Step {step}: {e.source_service} → {e.affected_service} ({e.new_status})")

    print("\nCascade chains:", sim.get_cascade_chain(mesh))
    print("Root cause for payment-service:", sim.get_root_cause_for("payment-service"))
