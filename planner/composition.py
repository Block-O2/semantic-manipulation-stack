"""Small symbolic state model for trusted manipulation-skill composition."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from planner.registry import SkillRegistry
from planner.schema import Goal, PlanStep
from world import WorldState


@dataclass
class SymbolicManipulationState:
    """Planner-only projection of facts changed by Pick, Place, and Push."""

    base_values: dict[str, object]
    object_names: tuple[str, ...]
    target_names: tuple[str, ...]
    push_region_names: tuple[str, ...]
    holding: str | None
    grasped: set[str]
    occupants: dict[str, tuple[str, ...]]
    unknown_occupied_targets: set[str]
    target_roles: dict[str, str]
    push_memberships: set[tuple[str, str]]

    @classmethod
    def from_world_state(cls, state: WorldState) -> "SymbolicManipulationState":
        return cls(
            base_values=state.flattened(),
            object_names=tuple(sorted(state.objects)),
            target_names=tuple(sorted(state.targets)),
            push_region_names=tuple(sorted(state.push_regions)),
            holding=state.robot.holding,
            grasped={name for name, value in state.objects.items() if value.grasped},
            occupants={
                name: tuple(sorted(value.occupied_by))
                for name, value in state.targets.items()
            },
            unknown_occupied_targets={
                name
                for name, value in state.targets.items()
                if value.occupied and not value.occupied_by
            },
            target_roles={name: value.role for name, value in state.targets.items()},
            push_memberships={
                (object_name, region_name)
                for object_name in state.objects
                for region_name in state.push_regions
                if state.relations.get(
                    WorldState.push_region_key(object_name, region_name),
                    False,
                )
            },
        )

    def flattened(self) -> dict[str, object]:
        values = dict(self.base_values)
        values["robot.holding"] = self.holding
        for object_name in self.object_names:
            values[f"objects.{object_name}.grasped"] = object_name in self.grasped
        for target_name in self.target_names:
            occupied_by = self.occupants[target_name]
            values[f"targets.{target_name}.occupied"] = bool(occupied_by) or (
                target_name in self.unknown_occupied_targets
            )
            values[f"targets.{target_name}.occupied_by"] = occupied_by
            for object_name in self.object_names:
                values[
                    f"relations.{WorldState.relation_key(object_name, target_name)}"
                ] = object_name in occupied_by
        for object_name in self.object_names:
            for region_name in self.push_region_names:
                values[
                    f"relations.{WorldState.push_region_key(object_name, region_name)}"
                ] = (object_name, region_name) in self.push_memberships
        return values

    def signature(self) -> tuple[object, ...]:
        return (
            self.holding,
            tuple(sorted(self.grasped)),
            tuple(
                (target, self.occupants[target]) for target in self.target_names
            ),
            tuple(sorted(self.unknown_occupied_targets)),
            tuple(sorted(self.push_memberships)),
        )

    def apply(self, step: PlanStep) -> "SymbolicManipulationState":
        occupants = dict(self.occupants)
        grasped = set(self.grasped)
        holding = self.holding
        push_memberships = set(self.push_memberships)

        if step.skill == "pick":
            object_name = step.args["object"]
            occupants = {
                target: tuple(item for item in names if item != object_name)
                for target, names in occupants.items()
            }
            holding = object_name
            grasped.add(object_name)
            push_memberships = {
                membership
                for membership in push_memberships
                if membership[0] != object_name
            }
        elif step.skill == "place":
            object_name = step.args["object"]
            target_name = step.args["target"]
            occupants = {
                target: tuple(item for item in names if item != object_name)
                for target, names in occupants.items()
            }
            occupants[target_name] = tuple(
                sorted((*occupants[target_name], object_name))
            )
            holding = None
            grasped.discard(object_name)
            push_memberships = {
                membership
                for membership in push_memberships
                if membership[0] != object_name
            }
        elif step.skill == "push":
            object_name = step.args["object"]
            region_name = step.args["target"]
            occupants = {
                target: tuple(item for item in names if item != object_name)
                for target, names in occupants.items()
            }
            push_memberships = {
                membership
                for membership in push_memberships
                if membership[0] != object_name
            }
            push_memberships.add((object_name, region_name))
        else:
            raise ValueError(f"no symbolic transition for skill {step.skill!r}")

        return SymbolicManipulationState(
            base_values=self.base_values,
            object_names=self.object_names,
            target_names=self.target_names,
            push_region_names=self.push_region_names,
            holding=holding,
            grasped=grasped,
            occupants=occupants,
            unknown_occupied_targets=set(self.unknown_occupied_targets),
            target_roles=self.target_roles,
            push_memberships=push_memberships,
        )

    def applicable_steps(self, registry: SkillRegistry) -> tuple[PlanStep, ...]:
        candidates: list[PlanStep] = []
        values = self.flattened()
        if self.holding is None and registry.get("pick") is not None:
            for object_name in self.object_names:
                step = PlanStep("pick", {"object": object_name})
                if not registry.check_preconditions(step, values):
                    candidates.append(step)
        elif self.holding is not None and registry.get("place") is not None:
            ordered_targets = sorted(
                self.target_names,
                key=lambda name: (
                    self.target_roles.get(name) != "temporary",
                    name,
                ),
            )
            for target_name in ordered_targets:
                step = PlanStep(
                    "place",
                    {"object": self.holding, "target": target_name},
                )
                if not registry.check_preconditions(step, values):
                    candidates.append(step)
        return tuple(candidates)


def compose_inside_plan(
    goal: Goal,
    world_state: WorldState,
    registry: SkillRegistry,
    *,
    max_depth: int = 6,
) -> tuple[PlanStep, ...] | None:
    """Breadth-first search over legal Pick / Place transitions."""

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    initial = SymbolicManipulationState.from_world_state(world_state)

    def satisfied(state: SymbolicManipulationState) -> bool:
        return goal.object_name in state.occupants.get(goal.target_name, ())

    if satisfied(initial):
        return ()

    queue = deque([(initial, ())])
    visited = {initial.signature()}
    while queue:
        state, steps = queue.popleft()
        if len(steps) >= max_depth:
            continue
        for step in state.applicable_steps(registry):
            successor = state.apply(step)
            successor_steps = (*steps, step)
            if satisfied(successor):
                return successor_steps
            signature = successor.signature()
            if signature not in visited:
                visited.add(signature)
                queue.append((successor, successor_steps))
    return None


def compose_push_plan(
    goal: Goal,
    world_state: WorldState,
    registry: SkillRegistry,
) -> tuple[PlanStep, ...] | None:
    """Return one valid semantic Push transition when the registry supports it."""

    if registry.get("push") is None:
        return None
    state = SymbolicManipulationState.from_world_state(world_state)
    if (goal.object_name, goal.target_name) in state.push_memberships:
        return ()
    step = PlanStep(
        "push",
        {"object": goal.object_name, "target": goal.target_name},
    )
    if registry.check_preconditions(step, state.flattened()):
        return None
    successor = state.apply(step)
    if (goal.object_name, goal.target_name) not in successor.push_memberships:
        return None
    return (step,)
