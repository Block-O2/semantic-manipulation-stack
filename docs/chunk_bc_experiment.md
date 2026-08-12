# Push Temporal Aliasing Diagnosis and Chunk BC

This is an exploratory engineering comparison on the existing 50-episode,
32,652-timestep, 20 Hz classical Push dataset. It keeps the semantic interface,
controller, episode split, normalization rule, optimizer family, and safety
limits fixed. It does not implement temporal ensembling or ACT.

Exact machine-readable outputs are under `artifacts/chunk_bc/`:

- `dataset_diagnostics.json`: motion, transition, aliasing, and stratified BC metrics;
- `offline_metrics.json`: Progress BC and K=10/20/40 Chunk BC metrics;
- `physical_comparison.json`: all matched physical trial records and summaries;
- `trajectories/`: every fifth learned control step for the first three trials;
- `trajectory_summary.json`: compact inspection summary.

Reproduction entry points:

```bash
python -m analysis.push_dataset_diagnostics \
  --dataset data/push_demos_50_seed123.npz \
  --checkpoint checkpoints/push_bc_seed17.npz \
  --output artifacts/chunk_bc/dataset_diagnostics.json

python -m learning.train_temporal_push_bc progress \
  --dataset data/push_demos_50_seed123.npz \
  --checkpoint checkpoints/push_progress_bc_seed17.npz --seed 17

python -m learning.train_temporal_push_bc chunk \
  --dataset data/push_demos_50_seed123.npz \
  --checkpoint checkpoints/push_chunk_k20_seed17.npz \
  --prediction-horizon 20 --seed 17

python -m evaluation.temporal_push_comparison \
  --dataset data/push_demos_50_seed123.npz \
  --one-step-checkpoint checkpoints/push_bc_seed17.npz \
  --progress-checkpoint checkpoints/push_progress_bc_seed17.npz \
  --chunk-checkpoint checkpoints/push_chunk_k20_seed17.npz \
  --execution-horizons 1 5 20 --trials 20 \
  --output artifacts/chunk_bc/physical_comparison.json \
  --trajectory-dir artifacts/chunk_bc/trajectories
```

The nearest-neighbor portion of diagnostics uses the optional `analysis`
dependency group (`pip install -e '.[analysis]'`).

## 1. Dataset diagnosis

For the absolute Cartesian action target, define:

```text
delta_a_t = ||a_t - a_(t-1)||
delta_a_0 = 0
STATIC when delta_a_t <= 0.003 m
MOVING when delta_a_t > 0.003 m
```

The 3 mm threshold was fixed before training. The empirical distribution has a
sub-millimetre static/jitter cluster, a gap, and moving target updates beginning
around 7 mm. It was not chosen from rollout outcomes.

| Statistic | Value |
|---|---:|
| min / median | 0 / 0 m |
| mean | 0.001335 m |
| p90 / p95 | 0.007305 / 0.007998 m |
| p99 / max | 0.029905 / 0.032991 m |
| STATIC target samples | 29,314 / 32,652 (89.78%) |
| MOVING target samples | 3,338 / 32,652 (10.22%) |
| per-episode STATIC ratio | 89.45% min / 89.77% mean / 90.04% max |

Important limitation: this classifies changes in the recorded absolute target,
not measured EE velocity. A target can remain constant for several control
steps while the trusted controller moves toward that waypoint. The ratio still
shows why overall regression metrics are dominated by repeated targets, but it
is not a literal claim that the physical robot is motionless 89.78% of the time.

### Transition regions

Raw target updates appear as pulses separated by controller convergence steps.
Pulses separated by at most 12 timesteps are therefore treated as one moving
segment. A transition event is either a debounced STATIC/MOVING boundary or a
direction change of at least 60 degrees between successive moving updates.
The transition mask is ±5 timesteps (±0.25 s at 20 Hz) around each event. It
contains 3,493 timesteps, or 10.70% of the dataset.

### State aliasing

Observations were standardized per feature over the dataset and indexed using
Euclidean nearest neighbors. With normalized observation distance <= 0.05 and
an action-distance materiality threshold of 1 cm:

- 12,093 unique close hold/move neighbor pairs were found;
- 7,265 of them differed in action by at least 1 cm;
- 7,080 material pairs are within one episode and 185 cross episode boundaries;
- episode 32, t=634 vs t=635 has observation distance `5.70e-7` but action
  distance `14.83 mm`;
- episode 49, t=34 vs t=35 has observation distance `1.71e-5` but action
  distance `30.00 mm`;
- cross-episode example episode 38 t=660 vs episode 31 t=571 has observation
  distance `0.00393` but action distance `13.35 mm`.

This is direct evidence that the 10D state does not uniquely identify the
expert's temporal intent. It does not prove aliasing is the only closed-loop
failure source.

## 2. Existing one-step BC

The existing checkpoint and its original validation episodes were reused.

| Validation subset | Samples | MSE (m²) | L1 |
|---|---:|---:|---:|
| Overall | 6,776 | 1.5478e-5 | 2.731 mm |
| STATIC target | 6,088 | 1.3282e-5 | 2.588 mm |
| MOVING target | 688 | 3.4914e-5 | 3.998 mm |
| Transition | 715 | 2.1772e-5 | 3.195 mm |

The small overall number is pulled toward the 89.8% repeated-target majority.
Moving-target L1 is 55% higher than hold-target L1. On the same 20 replay-stable
initial states, one-step BC remains 0/20: 13 `POLICY_TIMEOUT`, 7
`POLICY_ACTION_UNSAFE`, mean 520.95 control steps, and mean cube displacement
4.05 mm.

## 3. Progress-conditioned one-step BC

The only new input is `tau=t/(T-1)`. Architecture and training budget remain:

```text
11 → 64 ReLU → 64 ReLU → 3
60 epochs, batch 256, Adam-family NumPy optimizer, seed 17
```

The runtime advances tau over the existing 650 learned-control-step budget,
close to the dataset mean episode length of 653.04; it receives no classical
phase label.

| Validation subset | MSE (m²) | L1 |
|---|---:|---:|
| Overall | 1.5474e-5 | 2.841 mm |
| STATIC target | 1.3661e-5 | 2.706 mm |
| MOVING target | 3.1511e-5 | 4.032 mm |
| Transition | 1.8926e-5 | 3.124 mm |

Progress slightly reduces transition MSE but does not improve overall or
moving L1. Physical success is 0/20: 19 unsafe rejections and one timeout.
Mean displacement rises to 43.70 mm, so temporal position often makes the
policy start moving, but it does not produce a safe, target-reaching trajectory.

## 4. Deterministic Chunk BC

All chunk policies use the same architecture family and split:

```text
10 → 128 ReLU → 128 ReLU → 3K
masked MSE over valid, episode-bounded future actions
60 epochs, batch 256, seed 17
```

Padding repeats the final action but receives `mask=False`; it contributes no
gradient or metric. Normalization is fitted only on the 40 training episodes.

| K | Duration | First-action L1 | Masked chunk L1 | Masked MSE |
|---:|---:|---:|---:|---:|
| 10 | 0.5 s | 2.743 mm | 2.933 mm | 1.9790e-5 |
| 20 | 1.0 s | **2.506 mm** | 3.190 mm | 2.5240e-5 |
| 40 | 2.0 s | 2.979 mm | 3.556 mm | 3.4105e-5 |

Horizon-wise L1 trends upward. Representative values:

| K | k=0 | k=5 | k=9 | k=19 | k=39 |
|---:|---:|---:|---:|---:|---:|
| 10 | 2.743 | 2.909 | 3.097 | — | — |
| 20 | 2.506 | 2.952 | 2.853 | 3.748 | — |
| 40 | 2.979 | 2.984 | 3.481 | 3.778 | 3.987 |

K=10 has the best aggregate chunk error; K=20 has the best first action and a
useful 1 s prediction window. There is no unambiguous offline winner, so the
predeclared default K=20 was used for the H comparison. Full per-horizon arrays
are preserved in `offline_metrics.json`.

## 5. Matched physical comparison

The same 20 replay-stable dataset states were used for every backend. Four
additional dataset states were skipped before policy execution because reset
settling did not reproduce the stored position within 3 mm.

| Policy | K | H | Success | Failures | Mean steps | Mean displacement |
|---|---:|---:|---:|---|---:|---:|
| Classical | — | — | 20/20 | — | 658.70 | 240.27 mm |
| One-step BC | 1 | 1 | 0/20 | 13 timeout, 7 unsafe | 520.95 | 4.05 mm |
| Progress BC | 1 | 1 | 0/20 | 1 timeout, 19 unsafe | 425.40 | 43.70 mm |
| Chunk BC | 20 | 1 | 0/20 | 5 timeout, 15 unsafe | 270.35 | 11.17 mm |
| Chunk BC | 20 | 5 | 0/20 | 20 unsafe | 99.25 | ~0 mm |
| Chunk BC | 20 | 20 | **2/20** | 6 timeout, 12 unsafe | 499.50 | 41.90 mm |

Every learned action used the unchanged hard-rejection, max-step, workspace,
Cartesian primitive, controller, and MuJoCo path. Unsafe counts are outcomes,
not post-hoc filtering.

Trajectory inspection explains part of the H behavior:

- one-step frequently predicts small corrections and stays near the initial
  state; first-three-trial mean predicted step magnitude is roughly 3–10 mm;
- Progress often begins moving later, but accumulated distribution shift leads
  to a large unsafe target;
- H=1 repeatedly re-predicts k=0 and usually remains reactive or becomes unsafe;
- H=5 commits briefly, then re-observes and restarts at k=0; all 20 runs were
  rejected early and the cube did not move;
- H=20 preserves within-chunk temporal order long enough to move the cube and
  succeeds twice, but open-loop error and later re-planning still cause 18
  failures. It is evidence for temporal commitment, not a robust policy.

There is no evidence of a useful intermediate H=5 compromise in this setup.

## 6. Interpretation and stopping decision

The milestone supports a mixed diagnosis:

1. **Temporal aliasing is real.** Near-identical 10D states have materially
   different actions, and the target-static majority dominates overall MSE.
2. **A scalar clock is insufficient.** Progress changes behavior and increases
   displacement, but remains 0/20 and mostly unsafe.
3. **Chunking partially fixes “stay still.”** Full K=H=20 reaches 2/20 and moves
   farther than one-step, showing that coherent commitment can help.
4. **Distribution shift and compounding error remain dominant.** Far-horizon
   offline error grows, unsafe rejection remains common, and 18/20 full-chunk
   runs fail.
5. **H is not monotonic.** H=5 is worse than both H=1 and H=20 here because
   short re-planning repeatedly resets to the predicted chunk's first-action
   regime without preserving enough temporal order.

Temporal ensembling is not justified as the automatic next implementation:
the selected chunk policy is still only 2/20 and averaging overlapping biased
predictions may smooth commands without resolving missing intent or
out-of-distribution states. ACT is also not yet justified merely by these
results; a larger model does not remove the observation ambiguity. A future
milestone should first decide whether to add a minimal history/phase-like state
or recollect targets with a less aliased command representation, then compare
against this frozen result. This milestone stops here as required.
