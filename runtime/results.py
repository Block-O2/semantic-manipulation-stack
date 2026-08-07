"""Structured outcome for one deterministic sequence of semantic skills."""

from __future__ import annotations

from dataclasses import dataclass

from skills import SkillFailure, SkillResult


@dataclass(frozen=True)
class TaskResult:
    success: bool
    completed_steps: int
    failed_step: int | None
    failed_skill: str | None
    reason: SkillFailure | None
    skill_results: tuple[SkillResult, ...]
    trace: tuple[str, ...]
