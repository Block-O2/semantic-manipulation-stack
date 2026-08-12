"""Small configuration boundary for selecting a physical Push backend."""

from __future__ import annotations

from skills import (
    ACTPushBackend,
    BCPushBackend,
    ChunkBCPushBackend,
    ClassicalPushBackend,
    ProgressBCPushBackend,
    PushBackend,
)


def create_push_backend(
    name: str,
    *,
    checkpoint: str | None = None,
    execution_horizon: int | None = None,
    device: str = "auto",
    act_execution_mode: str = "queue",
    temporal_ensemble_coeff: float = 0.01,
) -> PushBackend:
    normalized = name.strip().lower()
    if normalized == "classical":
        if checkpoint is not None or execution_horizon is not None:
            raise ValueError("classical Push backend does not use learned-policy options")
        return ClassicalPushBackend()
    if normalized == "bc":
        if checkpoint is None:
            raise ValueError("BC Push backend requires a checkpoint")
        if execution_horizon is not None:
            raise ValueError("one-step BC does not use an execution horizon")
        return BCPushBackend.from_checkpoint(checkpoint)
    if normalized == "progress_bc":
        if checkpoint is None:
            raise ValueError("Progress BC Push backend requires a checkpoint")
        if execution_horizon is not None:
            raise ValueError("Progress BC does not use an execution horizon")
        return ProgressBCPushBackend.from_checkpoint(checkpoint)
    if normalized == "chunk_bc":
        if checkpoint is None or execution_horizon is None:
            raise ValueError("Chunk BC requires a checkpoint and execution horizon")
        return ChunkBCPushBackend.from_checkpoint(
            checkpoint, execution_horizon=execution_horizon
        )
    if normalized == "act":
        if checkpoint is None:
            raise ValueError("ACT Push backend requires a checkpoint")
        if execution_horizon is not None:
            raise ValueError("ACT uses checkpoint n_action_steps, not a runtime horizon")
        return ACTPushBackend.from_checkpoint(
            checkpoint,
            device=device,
            execution_mode=act_execution_mode,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
        )
    raise ValueError(f"unknown Push backend {name!r}")
