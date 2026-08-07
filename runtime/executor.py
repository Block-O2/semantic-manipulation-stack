"""Minimal ordered executor that stops at the first failed semantic skill."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from runtime.results import TaskResult
from skills import Skill, SkillResult


class SkillExecutor:
    """Execute caller-provided skills in order without planning or recovery."""

    def __init__(self, *, logger: Callable[[str], None] | None = None) -> None:
        self._logger = logger

    def _emit(self, trace: list[str], message: str) -> None:
        trace.append(message)
        if self._logger is not None:
            self._logger(message)

    def execute(self, skills: Iterable[Skill]) -> TaskResult:
        trace: list[str] = []
        results: list[SkillResult] = []
        self._emit(trace, "TaskExecutor")

        for step, skill in enumerate(skills):
            skill_name = type(skill).__name__
            self._emit(trace, f"STEP {step}: {skill_name}")
            result = skill.execute()
            results.append(result)
            if not result.success:
                self._emit(trace, f"TASK_FAILED: {result.reason.value if result.reason else 'UNKNOWN'}")
                return TaskResult(
                    success=False,
                    completed_steps=step,
                    failed_step=step,
                    failed_skill=result.skill,
                    reason=result.reason,
                    skill_results=tuple(results),
                    trace=tuple(trace),
                )

        self._emit(trace, "TASK_SUCCESS")
        return TaskResult(
            success=True,
            completed_steps=len(results),
            failed_step=None,
            failed_skill=None,
            reason=None,
            skill_results=tuple(results),
            trace=tuple(trace),
        )
