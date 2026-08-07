"""Planner abstraction, deterministic implementations, and optional LLM adapter."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from planner.registry import SkillRegistry
from planner.schema import Goal, Plan, PlannerFailureContext, PlannerResult, PlanStep
from world import WorldState


class PlannerBackend(ABC):
    @abstractmethod
    def plan(
        self,
        goal: Goal,
        world_state: WorldState,
        skill_registry: SkillRegistry,
        previous_plan: Plan | None = None,
        failure_context: PlannerFailureContext | None = None,
    ) -> PlannerResult:
        """Return structured planner output without executing anything."""


class RuleBasedPlanner(PlannerBackend):
    """Offline deterministic planner for tests and baseline evaluation."""

    def __init__(self) -> None:
        self.calls = 0

    def plan(
        self,
        goal: Goal,
        world_state: WorldState,
        skill_registry: SkillRegistry,
        previous_plan: Plan | None = None,
        failure_context: PlannerFailureContext | None = None,
    ) -> PlannerResult:
        self.calls += 1
        object_state = world_state.objects.get(goal.object_name)
        target_state = world_state.targets.get(goal.target_name)
        if object_state is None or not object_state.exists:
            return PlannerResult.from_plan(Plan.cannot_plan("Requested object does not exist."))
        if target_state is None or not target_state.exists:
            return PlannerResult.from_plan(Plan.cannot_plan("Requested target does not exist."))
        if not object_state.reachable:
            return PlannerResult.from_plan(Plan.cannot_plan("Requested object is unreachable."))
        if not target_state.reachable:
            return PlannerResult.from_plan(Plan.cannot_plan("Requested target is unreachable."))
        holding = world_state.robot.holding
        if holding not in {None, goal.object_name}:
            return PlannerResult.from_plan(
                Plan.cannot_plan("The gripper holds another object and no release skill is available.")
            )
        if holding == goal.object_name:
            steps = [
                PlanStep(
                    "place",
                    {"object": goal.object_name, "target": goal.target_name},
                )
            ]
        else:
            steps = [
                PlanStep("pick", {"object": goal.object_name}),
                PlanStep(
                    "place",
                    {"object": goal.object_name, "target": goal.target_name},
                ),
            ]
        return PlannerResult.from_plan(Plan.ready(steps))


class ScriptedPlanner(PlannerBackend):
    """Queue-backed mock planner for deterministic unit tests."""

    def __init__(self, responses: Sequence[PlannerResult | Plan | dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def plan(
        self,
        goal: Goal,
        world_state: WorldState,
        skill_registry: SkillRegistry,
        previous_plan: Plan | None = None,
        failure_context: PlannerFailureContext | None = None,
    ) -> PlannerResult:
        self.calls.append(
            {
                "goal": goal.to_dict(),
                "world_state": world_state.to_dict(),
                "previous_plan": previous_plan.to_dict() if previous_plan else None,
                "failure_context": failure_context.to_dict() if failure_context else None,
            }
        )
        if not self._responses:
            return PlannerResult.from_plan(Plan.cannot_plan("No scripted response remains."))
        response = self._responses.pop(0)
        if isinstance(response, PlannerResult):
            return response
        if isinstance(response, Plan):
            return PlannerResult.from_plan(response)
        return PlannerResult(raw_response=response)


class OpenAICompatiblePlanner(PlannerBackend):
    """Optional isolated JSON planner using an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("SEMANTIC_AGENT_MODEL", "")
        self.api_key = api_key or os.environ.get("SEMANTIC_AGENT_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("SEMANTIC_AGENT_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        if not self.model or not self.api_key:
            raise ValueError(
                "SEMANTIC_AGENT_MODEL and SEMANTIC_AGENT_API_KEY are required for LLM planning"
            )

    @staticmethod
    def _prompt_payload(
        goal: Goal,
        world_state: WorldState,
        skill_registry: SkillRegistry,
        previous_plan: Plan | None,
        failure_context: PlannerFailureContext | None,
    ) -> dict[str, Any]:
        return {
            "original_goal": goal.to_dict(),
            "current_world_state": world_state.to_dict(),
            "available_skills": skill_registry.to_dict(),
            "previous_plan": previous_plan.to_dict() if previous_plan else None,
            "failure_context": failure_context.to_dict() if failure_context else None,
            "required_output": {
                "ready": {
                    "status": "ready",
                    "steps": [{"skill": "pick", "args": {"object": "red_cube"}}],
                },
                "cannot_plan": {
                    "status": "cannot_plan",
                    "reason": "semantic reason",
                },
            },
        }

    def plan(
        self,
        goal: Goal,
        world_state: WorldState,
        skill_registry: SkillRegistry,
        previous_plan: Plan | None = None,
        failure_context: PlannerFailureContext | None = None,
    ) -> PlannerResult:
        prompt = self._prompt_payload(
            goal,
            world_state,
            skill_registry,
            previous_plan,
            failure_context,
        )
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a semantic robot task planner. Use only the supplied "
                        "skill registry. Return exactly one JSON object matching one of "
                        "the required output schemas. Never emit code or low-level commands."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, sort_keys=True)},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            decoded = json.loads(content)
            return PlannerResult(raw_response=decoded)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return PlannerResult(raw_response=None, error=f"invalid LLM response: {exc}")
        except (urllib.error.URLError, TimeoutError) as exc:
            return PlannerResult(raw_response=None, error=f"planner request failed: {exc}")
