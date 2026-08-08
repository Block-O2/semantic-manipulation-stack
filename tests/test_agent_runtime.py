from __future__ import annotations

from dataclasses import replace

from planner import (
    Goal,
    Plan,
    PlanStep,
    PlanValidator,
    RuleBasedPlanner,
    ScriptedPlanner,
    SkillRegistry,
)
from runtime import (
    AgentBoundaryEvent,
    AgentFailure,
    AgentRuntime,
    AgentStepEvent,
    SkillExecutor,
)
from skills import Skill, SkillPhase, SkillResult
from world import ObjectState, PoseState, RobotState, TargetState, WorldState


GOAL = Goal.put_inside(
    "Put the red cube inside the blue target.",
    "red_cube",
    "blue_target",
)


class FakeWorldMachine:
    def __init__(
        self,
        *,
        holding: str | None = None,
        inside: bool = False,
        object_reachable: bool = True,
        target_reachable: bool = True,
    ) -> None:
        pose = PoseState((0.0, 0.0, 0.825), (0.0, 0.0, 0.0, 1.0))
        self.state = WorldState(
            robot=RobotState(holding),
            objects={
                "red_cube": ObjectState(
                    True, pose, holding == "red_cube", object_reachable
                )
            },
            targets={
                "blue_target": TargetState(True, pose, target_reachable, inside)
            },
            relations={"red_cube_inside_blue_target": inside},
        )

    def observe(self) -> WorldState:
        return self.state

    def pick(self) -> None:
        obj = replace(self.state.objects["red_cube"], grasped=True)
        self.state = replace(
            self.state,
            robot=RobotState("red_cube"),
            objects={"red_cube": obj},
        )

    def place(self) -> None:
        obj = replace(self.state.objects["red_cube"], grasped=False)
        target = replace(
            self.state.targets["blue_target"],
            occupied=True,
            occupied_by=("red_cube",),
        )
        self.state = replace(
            self.state,
            robot=RobotState(None),
            objects={"red_cube": obj},
            targets={"blue_target": target},
            relations={"red_cube_inside_blue_target": True},
        )

    def lose_object(self) -> None:
        obj = replace(self.state.objects["red_cube"], grasped=False, reachable=True)
        self.state = replace(
            self.state,
            robot=RobotState(None),
            objects={"red_cube": obj},
        )

    def set_target_reachable(self, reachable: bool) -> None:
        target = replace(
            self.state.targets["blue_target"],
            reachable=reachable,
            pose=PoseState(
                (0.20, 0.10, 0.803) if reachable else (0.50, 0.50, 0.803),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        self.state = replace(self.state, targets={"blue_target": target})

    def occupy_target(self) -> None:
        target = replace(self.state.targets["blue_target"], occupied=True)
        self.state = replace(self.state, targets={"blue_target": target})

    def remove_object(self) -> None:
        obj = replace(
            self.state.objects["red_cube"],
            exists=False,
            pose=None,
            grasped=False,
            reachable=False,
        )
        self.state = replace(
            self.state,
            robot=RobotState(None),
            objects={"red_cube": obj},
        )


def success_result(skill: str) -> SkillResult:
    return SkillResult(
        success=True,
        skill=skill,
        object_name="red_cube",
        phase=SkillPhase.SUCCESS,
        reason=None,
        attempts=1,
        primitive_result=None,
        trace=(SkillPhase.SUCCESS.value,),
    )


class StateChangingSkill(Skill):
    def __init__(self, name: str, effect) -> None:
        self.name = name
        self.effect = effect

    def check_preconditions(self) -> SkillResult | None:
        return None

    def execute(self) -> SkillResult:
        self.effect()
        return success_result(self.name)


class FakeSkillFactory:
    def __init__(self, machine: FakeWorldMachine, *, suppress_effects: bool = False) -> None:
        self.machine = machine
        self.suppress_effects = suppress_effects
        self.created: list[str] = []

    def create(self, step: PlanStep) -> Skill:
        self.created.append(step.skill)
        if self.suppress_effects:
            return StateChangingSkill(step.skill.title() + "Skill", lambda: None)
        effect = self.machine.pick if step.skill == "pick" else self.machine.place
        return StateChangingSkill(step.skill.title() + "Skill", effect)


def runtime_for(machine: FakeWorldMachine, planner, *, max_replans: int = 2, suppress=False):
    registry = SkillRegistry.standard()
    return AgentRuntime(
        planner=planner,
        registry=registry,
        validator=PlanValidator(registry),
        observer=machine.observe,
        skill_factory=FakeSkillFactory(machine, suppress_effects=suppress),
        executor=SkillExecutor(),
        max_replans=max_replans,
    )


def test_nominal_agent_uses_one_plan_and_zero_replans() -> None:
    machine = FakeWorldMachine()
    planner = RuleBasedPlanner()

    result = runtime_for(machine, planner).run(GOAL)

    assert result.success
    assert result.planner_calls == 1
    assert result.replans == 0
    assert len(result.executed_steps) == 2
    assert all(step.residual.consistent for step in result.executed_steps)


def test_object_loss_residual_triggers_planner_replan_and_recovers() -> None:
    machine = FakeWorldMachine()
    planner = RuleBasedPlanner()
    disturbed = False

    def lose_after_first_pick(event: AgentStepEvent) -> None:
        nonlocal disturbed
        if not disturbed and event.step.skill == "pick":
            machine.lose_object()
            disturbed = True

    result = runtime_for(machine, planner).run(GOAL, after_step=lose_after_first_pick)

    assert result.success
    assert result.planner_calls == 2
    assert result.replans == 1
    assert len(result.plan_history) == 2
    assert result.plan_history[1].steps[0].skill == "pick"
    assert not result.executed_steps[0].residual.consistent
    assert any("REPLAN REQUIRED: SEMANTIC_RESIDUAL" in line for line in result.trace)


def test_replanning_budget_exhaustion_stops_infinite_loop() -> None:
    machine = FakeWorldMachine()
    planner = RuleBasedPlanner()

    result = runtime_for(
        machine,
        planner,
        max_replans=1,
        suppress=True,
    ).run(GOAL)

    assert not result.success
    assert result.failure_reason is AgentFailure.AGENT_REPLAN_EXHAUSTED
    assert result.planner_calls == 2
    assert result.replans == 1


def test_cannot_plan_stops_cleanly_without_skill_execution() -> None:
    machine = FakeWorldMachine(target_reachable=False)
    planner = RuleBasedPlanner()

    result = runtime_for(machine, planner).run(GOAL)

    assert not result.success
    assert result.failure_reason is AgentFailure.CANNOT_PLAN
    assert result.planner_calls == 1
    assert result.replans == 0
    assert result.executed_steps == ()


def test_goal_satisfaction_stops_before_remaining_plan_steps() -> None:
    machine = FakeWorldMachine()
    plan = Plan.ready(
        [
            PlanStep("pick", {"object": "red_cube"}),
            PlanStep(
                "place", {"object": "red_cube", "target": "blue_target"}
            ),
            PlanStep("pick", {"object": "red_cube"}),
        ]
    )
    planner = ScriptedPlanner([plan])
    factory = FakeSkillFactory(machine)
    registry = SkillRegistry.standard()
    runtime = AgentRuntime(
        planner=planner,
        registry=registry,
        validator=PlanValidator(registry),
        observer=machine.observe,
        skill_factory=factory,
    )

    result = runtime.run(GOAL)

    assert result.success
    assert factory.created == ["pick", "place"]
    assert len(result.executed_steps) == 2


def test_invalid_plans_consume_replan_budget_then_stop() -> None:
    machine = FakeWorldMachine()
    planner = ScriptedPlanner(
        [
            {"steps": [{"skill": "push", "args": {"object": "red_cube"}}]},
            {"steps": [{"skill": "move_joints", "args": {}}]},
        ]
    )

    result = runtime_for(machine, planner, max_replans=1).run(GOAL)

    assert not result.success
    assert result.failure_reason is AgentFailure.AGENT_REPLAN_EXHAUSTED
    assert result.planner_calls == 2
    assert result.replans == 1


def test_reachable_target_move_is_reobserved_without_replanning() -> None:
    machine = FakeWorldMachine()
    planner = RuleBasedPlanner()
    moved = False

    def move_before_place(event: AgentBoundaryEvent) -> None:
        nonlocal moved
        if event.step.skill == "place" and not moved:
            machine.set_target_reachable(True)
            moved = True

    result = runtime_for(machine, planner).run(GOAL, before_step=move_before_place)

    assert result.success
    assert result.planner_calls == 1
    assert result.replans == 0
    assert result.executed_steps[1].world_before.targets[
        "blue_target"
    ].pose.position == (0.20, 0.10, 0.803)


def test_unreachable_target_move_triggers_precondition_replan() -> None:
    machine = FakeWorldMachine()
    planner = RuleBasedPlanner()
    moved = False

    def move_before_place(event: AgentBoundaryEvent) -> None:
        nonlocal moved
        if event.step.skill == "place" and not moved:
            machine.set_target_reachable(False)
            moved = True

    result = runtime_for(machine, planner).run(GOAL, before_step=move_before_place)

    assert not result.success
    assert result.failure_reason is AgentFailure.CANNOT_PLAN
    assert result.planner_calls == 2
    assert result.replans == 1
    assert any("TARGET_UNREACHABLE" in line for line in result.trace)


def test_unknown_target_occupant_causes_clean_cannot_plan_on_replan() -> None:
    machine = FakeWorldMachine()
    planner = RuleBasedPlanner()
    occupied = False

    def occupy_before_place(event: AgentBoundaryEvent) -> None:
        nonlocal occupied
        if event.step.skill == "place" and not occupied:
            machine.occupy_target()
            occupied = True

    result = runtime_for(machine, planner).run(GOAL, before_step=occupy_before_place)

    assert not result.success
    assert result.failure_reason is AgentFailure.CANNOT_PLAN
    assert result.replans == 1
    assert result.plan_history[-1].missing_capabilities == ()
    assert "occupying object is unknown" in (result.failure_detail or "")


def test_object_removal_is_observed_before_execution() -> None:
    machine = FakeWorldMachine()
    planner = RuleBasedPlanner()
    removed = False

    def remove_before_pick(event: AgentBoundaryEvent) -> None:
        nonlocal removed
        if not removed:
            machine.remove_object()
            removed = True

    result = runtime_for(machine, planner).run(GOAL, before_step=remove_before_pick)

    assert not result.success
    assert result.failure_reason is AgentFailure.CANNOT_PLAN
    assert result.executed_steps == ()
    assert result.replans == 1
    assert any("OBJECT_NOT_FOUND" in line for line in result.trace)


def test_open_drawer_goal_returns_explicit_capability_gap() -> None:
    machine = FakeWorldMachine()
    result = runtime_for(machine, RuleBasedPlanner()).run(
        Goal.open_drawer("Open the drawer.")
    )

    assert not result.success
    assert result.failure_reason is AgentFailure.CAPABILITY_GAP
    assert result.plan_history[-1].missing_capabilities == ("open_drawer",)
    assert result.executed_steps == ()
