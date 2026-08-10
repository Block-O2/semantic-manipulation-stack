"""One-to-one skill to task-specific ACT checkpoint routing."""

from __future__ import annotations

from pathlib import Path


class PolicyRouter:
    def __init__(self, checkpoints: dict[str, Path]) -> None:
        self._checkpoints = {skill: path.resolve() for skill, path in checkpoints.items()}

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._checkpoints))

    def checkpoint_for(self, skill: str) -> Path:
        try:
            checkpoint = self._checkpoints[skill]
        except KeyError as exc:
            raise KeyError(
                f"No policy for {skill!r}; available: {', '.join(self.capabilities)}"
            ) from exc
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ACT checkpoint does not exist: {checkpoint}")
        return checkpoint
