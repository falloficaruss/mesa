"""Synthetic logistics benchmark focused on entity indexing and memory pressure."""

from __future__ import annotations

from mesa import Agent, Model
from mesa.experimental.meta_agents.backend import MembershipBackend
from mesa.experimental.meta_agents.identity import ensure_entity_index
from mesa.experimental.meta_agents.meta_agent import MetaAgent, create_meta_agent
from mesa.experimental.scenarios import Scenario


class LogisticsScenario(Scenario):
    """Scenario parameters for the logistics benchmark workload."""

    hubs: int = 8
    crews: int = 128
    parcels: int = 1_024
    reassignments_per_step: int = 128
    refresh_batch: int = 64


class _TrackedAgent(Agent):
    """Agent helper that can clean up backend memberships on removal."""

    cleanup_mode = "agent"

    def remove(self) -> None:
        backend = getattr(self.model, "membership_backend", None)
        if backend is not None:
            if self.cleanup_mode in {"agent", "both"}:
                backend.remove_agent(self)
            if self.cleanup_mode in {"group", "both"}:
                backend.remove_group(self)
        super().remove()


class HubAgent(_TrackedAgent):
    """Distribution hub that anchors parcel and crew memberships."""

    cleanup_mode = "group"

    def __init__(self, model: Model, hub_code: int):
        super().__init__(model)
        self.hub_code = hub_code
        self.backlog = 0
        self.throughput = 0


class ParcelAgent(_TrackedAgent):
    """Shipment that moves between hubs and drives membership churn."""

    def __init__(
        self,
        model: Model,
        origin_hub: HubAgent,
        destination_hub: HubAgent,
        priority: int,
    ):
        super().__init__(model)
        self.origin_hub = origin_hub
        self.current_hub = origin_hub
        self.destination_hub = destination_hub
        self.priority = priority
        self.scan_count = 0


class RouterAgent(_TrackedAgent):
    """Crew role that directs parcel routing."""

    def __init__(self, model: Model, hub_code: int):
        super().__init__(model)
        self.hub_code = hub_code
        self.route_bias = 0


class ScannerAgent(_TrackedAgent):
    """Crew role that tracks parcel scans."""

    def __init__(self, model: Model, hub_code: int):
        super().__init__(model)
        self.hub_code = hub_code
        self.scan_total = 0


class LoaderAgent(_TrackedAgent):
    """Crew role that handles loading throughput."""

    def __init__(self, model: Model, hub_code: int):
        super().__init__(model)
        self.hub_code = hub_code
        self.loaded = 0


class LogisticsHubBenchmark(Model):
    """Large synthetic logistics simulation for benchmarking entity indexing."""

    def __init__(self, scenario: LogisticsScenario = LogisticsScenario):
        super().__init__(scenario=scenario)
        self.membership_backend = MembershipBackend()
        self.entity_index = ensure_entity_index(self)
        self.hubs = [HubAgent(self, hub_code=index) for index in range(scenario.hubs)]
        self.parcels: list[ParcelAgent] = []
        self.crews: list[Agent] = []
        self._parcel_cursor = 0
        self._crew_cursor = 0

        self._create_parcels(scenario)
        self._create_crews(scenario)

    def _create_parcels(self, scenario: LogisticsScenario) -> None:
        """Create a large parcel pool and register baseline memberships."""
        origin_hubs = self.rng.integers(0, scenario.hubs, size=scenario.parcels)
        destination_hubs = self.rng.integers(0, scenario.hubs, size=scenario.parcels)
        priorities = self.rng.integers(0, 5, size=scenario.parcels)

        for index in range(scenario.parcels):
            origin_hub = self.hubs[int(origin_hubs[index])]
            destination_hub = self.hubs[int(destination_hubs[index])]
            priority = int(priorities[index])
            parcel = ParcelAgent(self, origin_hub, destination_hub, priority)
            parcel.manifest_code = f"parcel-{parcel.entity_id}"
            self.parcels.append(parcel)

            self.membership_backend.add_membership(parcel, origin_hub, "queued")
            if priority >= 3:
                self.membership_backend.add_membership(parcel, origin_hub, "priority")
            origin_hub.backlog += 1

    def _create_crews(self, scenario: LogisticsScenario) -> None:
        """Create crew meta-agents with overlapping hub assignments."""
        for crew_id in range(scenario.crews):
            home_hub = self.hubs[crew_id % scenario.hubs]
            backup_hub = self.hubs[(crew_id + 1) % scenario.hubs]

            router = RouterAgent(self, home_hub.hub_code)
            scanner = ScannerAgent(self, home_hub.hub_code)
            loader = LoaderAgent(self, home_hub.hub_code)

            def remove_crew(crew):
                """Remove crew memberships even if teardown is partial."""
                try:
                    MetaAgent.remove(crew)
                finally:
                    self.membership_backend.remove_agent(crew)
                    self.membership_backend.remove_group(crew)

            crew = create_meta_agent(
                self,
                "LogisticsCrew",
                [router, scanner, loader],
                Agent,
                meta_attributes={
                    "crew_id": crew_id,
                    "home_hub": home_hub,
                    "backup_hub": backup_hub,
                    "status": "idle",
                },
                meta_methods={"remove": remove_crew},
            )

            if crew is None:
                continue

            self.crews.append(crew)
            self.membership_backend.bulk_add(
                [
                    (router, crew, "router"),
                    (scanner, crew, "scanner"),
                    (loader, crew, "loader"),
                    (crew, home_hub, "assigned"),
                    (crew, backup_hub, "backup"),
                ]
            )

    @staticmethod
    def _slice_window(items: list, start: int, count: int) -> list:
        """Return a wraparound slice from a list."""
        if not items:
            return []

        end = start + count
        if end <= len(items):
            return items[start:end]

        overflow = end - len(items)
        return items[start:] + items[:overflow]

    def _refresh_entity_directory(
        self, parcels: list[ParcelAgent], crews: list[Agent]
    ) -> None:
        """Repeatedly refresh explicit entity ids for a moving subset."""
        step_id = int(self.time)

        for offset, parcel in enumerate(parcels):
            parcel.unique_id = f"parcel-{parcel.entity_id}-step-{step_id}-{offset}"
            self.entity_index.register(parcel, kind="atomic")

        for offset, crew in enumerate(crews):
            crew.unique_id = f"crew-{crew.entity_id}-step-{step_id}-{offset}"
            self.entity_index.register(crew, kind="meta")
            for role in crew.agents:
                self.entity_index.register(role, kind="atomic")

    def _move_parcels(self, parcels: list[ParcelAgent]) -> None:
        """Move a subset of parcels to the next hub and churn memberships."""
        for offset, parcel in enumerate(parcels):
            old_hub = parcel.current_hub
            new_hub = self.hubs[(old_hub.hub_code + 1 + offset) % len(self.hubs)]

            if new_hub is old_hub:
                continue

            self.membership_backend.remove_membership(parcel, old_hub, "queued")
            self.membership_backend.remove_membership(parcel, old_hub, "priority")

            old_hub.backlog = max(0, old_hub.backlog - 1)
            new_hub.backlog += 1
            new_hub.throughput += 1

            parcel.current_hub = new_hub
            parcel.scan_count += 1

            self.membership_backend.add_membership(parcel, new_hub, "queued")
            if parcel.priority >= 3 or (offset + int(self.time)) % 4 == 0:
                self.membership_backend.add_membership(parcel, new_hub, "priority")

    def _rotate_crews(self, crews: list[Agent]) -> None:
        """Move a subset of crews between hub assignments."""
        for offset, crew in enumerate(crews):
            old_home = crew.home_hub
            new_home = self.hubs[(old_home.hub_code + 1 + offset) % len(self.hubs)]
            new_backup = self.hubs[(new_home.hub_code + 1) % len(self.hubs)]

            self.membership_backend.remove_membership(crew, old_home, "assigned")
            self.membership_backend.remove_membership(crew, crew.backup_hub, "backup")

            crew.home_hub = new_home
            crew.backup_hub = new_backup
            crew.status = "routing" if crew.status == "idle" else "idle"

            self.membership_backend.add_membership(crew, new_home, "assigned")
            self.membership_backend.add_membership(crew, new_backup, "backup")

    def step(self) -> None:
        """Run one dispatch wave and refresh entity ids."""
        if not self.parcels or not self.crews:
            return

        parcel_batch = self._slice_window(
            self.parcels,
            self._parcel_cursor,
            min(len(self.parcels), self.scenario.reassignments_per_step),
        )
        crew_batch = self._slice_window(
            self.crews,
            self._crew_cursor,
            min(len(self.crews), self.scenario.refresh_batch),
        )

        self._parcel_cursor = (self._parcel_cursor + len(parcel_batch)) % len(
            self.parcels
        )
        self._crew_cursor = (self._crew_cursor + len(crew_batch)) % len(self.crews)

        self._move_parcels(parcel_batch)
        self._rotate_crews(crew_batch)
        self._refresh_entity_directory(parcel_batch, crew_batch)
