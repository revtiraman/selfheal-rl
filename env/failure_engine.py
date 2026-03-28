"""Failure scenario generation — random and template-based failure patterns."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    DIFFICULTY_PRESETS,
    FAILURE_TYPES,
    SERVICES,
    SERVICE_NAMES,
)
from env.service_mesh import ServiceMesh


@dataclass
class Scenario:
    """A complete failure scenario."""

    difficulty: str
    root_failures: List[Tuple[str, str]]  # [(service, failure_type), ...]
    expected_fix_order: List[str]  # optimal order to fix services
    description: str = ""
    template_name: Optional[str] = None

    def __repr__(self) -> str:
        roots = ", ".join(f"{s}({ft})" for s, ft in self.root_failures)
        return f"Scenario({self.difficulty}, roots=[{roots}])"


# ─────────────────────────────────────────────
# Template scenarios (hand-designed, realistic)
# ─────────────────────────────────────────────

SCENARIO_TEMPLATES: Dict[str, dict] = {
    "database_cascade": {
        "difficulty": "MEDIUM",
        "root_failures": [("user-db", "disk_full")],
        "description": "user-db fills up → auth-service → api-gateway → payment-service all cascade",
        "expected_fix_order": ["user-db", "auth-service", "notification-service", "api-gateway", "payment-service"],
    },
    "cache_storm": {
        "difficulty": "MEDIUM",
        "root_failures": [("cache-layer", "memory_leak")],
        "description": "cache-layer OOMs → auth, search, order all lose their cache and degrade",
        "expected_fix_order": ["cache-layer", "auth-service", "search-service", "order-service"],
    },
    "deploy_gone_wrong": {
        "difficulty": "HARD",
        "root_failures": [("payment-service", "bad_deployment")],
        "description": "Bad deploy to payment-service → confusing errors, hard to diagnose",
        "expected_fix_order": ["payment-service"],
    },
    "double_trouble": {
        "difficulty": "HARD",
        "root_failures": [("user-db", "cpu_spike"), ("cache-layer", "memory_leak")],
        "description": "Two root services fail simultaneously → massive cascade",
        "expected_fix_order": [
            "user-db", "cache-layer", "auth-service", "notification-service",
            "search-service", "order-service", "api-gateway", "payment-service",
        ],
    },
    "slow_burn": {
        "difficulty": "MEDIUM",
        "root_failures": [("auth-service", "memory_leak")],
        "description": "Slow memory leak in auth degrades the entire system over 15 steps",
        "expected_fix_order": ["auth-service", "api-gateway", "payment-service"],
    },
    "flash_crash": {
        "difficulty": "EASY",
        "root_failures": [("api-gateway", "cpu_spike")],
        "description": "Sudden CPU spike in api-gateway — fast cascade to user-facing services",
        "expected_fix_order": ["api-gateway"],
    },
    "network_split": {
        "difficulty": "HARD",
        "root_failures": [("restaurant-db", "network_partition"), ("order-db", "network_partition")],
        "description": "Network partition isolates both databases — search and order services fail",
        "expected_fix_order": ["restaurant-db", "order-db", "search-service", "order-service", "payment-service"],
    },
}


class FailureEngine:
    """Generates failure scenarios of varying difficulty."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()

    def generate_scenario(self, difficulty: str = "MEDIUM") -> Scenario:
        """Generate a random failure scenario at the given difficulty."""
        preset = DIFFICULTY_PRESETS[difficulty]

        # Decide number of root failures
        num_roots = preset["num_root_failures"]
        if isinstance(num_roots, tuple):
            num_roots = self.rng.integers(num_roots[0], num_roots[1] + 1)

        # Pick root services (prefer services with dependents for bigger cascades)
        root_candidates = self._rank_by_impact()
        if difficulty == "EASY":
            # Easy: pick a leaf service or one with few dependents
            root_candidates = root_candidates[::-1]

        selected_roots: List[str] = []
        for candidate in root_candidates:
            if len(selected_roots) >= num_roots:
                break
            selected_roots.append(candidate)

        # Assign random failure types
        root_failures = [
            (svc, self.rng.choice(FAILURE_TYPES))
            for svc in selected_roots
        ]

        # Compute expected fix order (topological from roots outward)
        expected_order = self._compute_fix_order(selected_roots)

        return Scenario(
            difficulty=difficulty,
            root_failures=root_failures,
            expected_fix_order=expected_order,
            description=f"Random {difficulty} scenario: {len(selected_roots)} root failure(s)",
        )

    def generate_from_template(self, template_name: str) -> Scenario:
        """Generate a scenario from a predefined template."""
        tmpl = SCENARIO_TEMPLATES[template_name]
        return Scenario(
            difficulty=tmpl["difficulty"],
            root_failures=[(s, ft) for s, ft in tmpl["root_failures"]],
            expected_fix_order=list(tmpl["expected_fix_order"]),
            description=tmpl["description"],
            template_name=template_name,
        )

    def generate_chaos(self) -> Scenario:
        """Generate a chaotic scenario with staggered failures."""
        num_failures = self.rng.integers(2, 5)
        all_services = list(SERVICE_NAMES)
        self.rng.shuffle(all_services)
        selected = all_services[:num_failures]

        root_failures = [
            (svc, self.rng.choice(FAILURE_TYPES))
            for svc in selected
        ]

        expected_order = self._compute_fix_order(selected)

        return Scenario(
            difficulty="CHAOS",
            root_failures=root_failures,
            expected_fix_order=expected_order,
            description=f"Chaos scenario: {num_failures} random failures",
        )

    def apply_scenario(self, mesh: ServiceMesh, scenario: Scenario) -> None:
        """Inject the scenario's root failures into a service mesh."""
        for service_name, failure_type in scenario.root_failures:
            mesh.inject_failure(service_name, failure_type)

    # ── Helpers ───────────────────────────────────────────────

    def _rank_by_impact(self) -> List[str]:
        """Rank services by how many transitive dependents they have (most impact first)."""
        impact: Dict[str, int] = {}
        for name in SERVICE_NAMES:
            # BFS count of transitive dependents
            visited = set()
            queue = [name]
            while queue:
                current = queue.pop(0)
                for other_name, other_cfg in SERVICES.items():
                    if current in other_cfg["depends_on"] and other_name not in visited:
                        visited.add(other_name)
                        queue.append(other_name)
            impact[name] = len(visited)

        return sorted(impact.keys(), key=lambda x: impact[x], reverse=True)

    def _compute_fix_order(self, root_services: List[str]) -> List[str]:
        """Compute the optimal fix order: roots first, then dependents in topo order."""
        # Start with root services (sorted by impact — fix highest impact first)
        ranked = self._rank_by_impact()
        roots_ordered = [s for s in ranked if s in root_services]

        # Then add all transitively affected services
        affected = set()
        for root in root_services:
            self._collect_affected(root, affected)

        # Order affected services: dependencies before dependents
        dependents_ordered = [s for s in ranked if s in affected and s not in root_services]

        return roots_ordered + dependents_ordered

    def _collect_affected(self, service: str, affected: set) -> None:
        """BFS to find all transitively dependent services."""
        for name, cfg in SERVICES.items():
            if service in cfg["depends_on"] and name not in affected:
                affected.add(name)
                self._collect_affected(name, affected)


if __name__ == "__main__":
    engine = FailureEngine()
    mesh = ServiceMesh()

    # Random scenario
    scenario = engine.generate_scenario(difficulty="MEDIUM")
    print("Random scenario:", scenario)
    print("Expected fix order:", scenario.expected_fix_order)

    # Template scenario
    scenario_t = engine.generate_from_template("database_cascade")
    print("\nTemplate scenario:", scenario_t)
    print("Description:", scenario_t.description)

    # Apply and watch
    engine.apply_scenario(mesh, scenario_t)
    for step in range(10):
        events = mesh.tick()
        down = mesh.get_down_services()
        degraded = mesh.get_degraded_services()
        print(f"  Step {step}: DOWN={down}, DEGRADED={degraded}")
