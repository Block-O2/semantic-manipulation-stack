# Experiments Index

This index separates current meaningful runtime paths from retained diagnostic
baselines. All learned Push experiments preserve the same semantic
`Push(object, target)` interface and trusted Cartesian controller boundary.

## Results

| Experiment | Role | Main matched result | Detailed report |
|---|---|---:|---|
| Classical Push | Current deterministic baseline | 20/20 | [push_skill.md](push_skill.md) |
| One-step BC | Naive imitation diagnostic | 0/20 | [learned_push_backend.md](learned_push_backend.md) |
| Progress BC | Scalar-progress diagnostic | 0/20 | [chunk_bc_experiment.md](chunk_bc_experiment.md) |
| Chunk BC K=20/H=20 | Action-chunk diagnostic | 2/20 | [chunk_bc_experiment.md](chunk_bc_experiment.md) |
| LeRobot ACT queue K=32/H=8 | Mature ACT queue reproduction | 0/20 | [act_push_backend.md](act_push_backend.md) |
| ACT native Temporal Ensemble K=32/H=1 | Current learned Push mode | 20/20 | [act_push_backend.md](act_push_backend.md) |
| State Diffusion Policy To=2/Tp=16/Ta=8 | Current learned Push mode | 20/20 | [diffusion_push_backend.md](diffusion_push_backend.md) |

The ACT queue and Temporal Ensemble rows use the same checkpoint weights. Only
the inference mechanism changes. These are fixed-workspace, replay-stable
MuJoCo comparisons—not open-scene or real-robot validation.

## Evidence and local inputs

Committed, lightweight evidence is stored under:

```text
artifacts/chunk_bc/
artifacts/act_push/
artifacts/act_temporal_ensemble/
artifacts/diffusion_push/
```

These directories contain JSON metrics, physical comparison records, and
representative trajectories. Training datasets (`data/*.npz`) and learned
checkpoints (`checkpoints/*.npz`, `*.pt`, and associated local metadata) are
Git ignored. Therefore the committed reports are directly inspectable, while
rerunning learned demos requires locally generated inputs.

## Reproduction entry points

```bash
# Deterministic nominal checks
python -m evaluation.pick_place_trials --trials 20 --seed 47
python -m evaluation.agent_trials --trials 20 --seed 59
python -m evaluation.push_trials --trials 20 --seed 127

# Demonstration dataset
python -m datasets.collect_push_demos \
  --episodes 50 --seed 123 --output data/push_demos_50_seed123.npz

# One-step and Chunk BC details/commands
# See docs/learned_push_backend.md and docs/chunk_bc_experiment.md

# ACT training, queue evaluation, and Temporal Ensemble evaluation
# See docs/act_push_backend.md

# Diffusion Policy training and single frozen evaluation
# See docs/diffusion_push_backend.md
```

The diagnostic modes remain available for reproducibility, but they are not
presented as recommended robot runtimes.
