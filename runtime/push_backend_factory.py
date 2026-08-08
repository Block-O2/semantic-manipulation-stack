"""Small configuration boundary for selecting a physical Push backend."""

from __future__ import annotations

from skills import BCPushBackend, ClassicalPushBackend, PushBackend


def create_push_backend(
    name: str,
    *,
    checkpoint: str | None = None,
) -> PushBackend:
    normalized = name.strip().lower()
    if normalized == "classical":
        if checkpoint is not None:
            raise ValueError("classical Push backend does not use a checkpoint")
        return ClassicalPushBackend()
    if normalized == "bc":
        if checkpoint is None:
            raise ValueError("BC Push backend requires a checkpoint")
        return BCPushBackend.from_checkpoint(checkpoint)
    raise ValueError(f"unknown Push backend {name!r}")
