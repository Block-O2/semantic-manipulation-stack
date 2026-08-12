# Standard LeRobot ACT Push Backend

This milestone freezes the existing state/action/data interface and asks one
narrow question: does a maintained ACT implementation solve Push where the
one-step and simple MLP chunk baselines did not?

```text
ClassicalPushBackend
→ 50 successful demonstrations at 20 Hz
→ Hugging Face LeRobot ACTPolicy 0.4.4
→ ACTPushBackend
→ existing PushSkill semantic verification
→ existing trusted Cartesian controller
→ MuJoCo
```

No previous action, progress, phase, image, delta action, expert fallback, or
scripted trajectory is supplied to ACT.

## Data contract

The existing pickle-free NPZ is used without recollection: 50 episodes and
32,652 timesteps. The only meaningful input is the original 10D world-frame
state `[ee_xyz, cube_xyz, target_lower_xy, target_upper_xy]`. The output is the
original 3D absolute desired EE xyz in metres. The fixed seed-17 split is 40
train / 10 validation episodes; normalization uses only training episodes.
Future chunks repeat the final episode action for storage but mark those
positions as padding, so they never cross an episode.

LeRobot 0.4.4 requires an image or environment-state feature even when a robot
state is present. The adapter therefore supplies an empty `(0,)`
`observation.environment_state` compatibility tensor. It contains no value and
adds no observation information; the effective input remains exactly the 10D
`observation.state`. A narrow import shim is also required because this
repository's historical top-level `datasets` package otherwise shadows
Hugging Face `datasets`, which LeRobot imports internally.

## Fixed ACT configuration and training

One configuration was trained, with no sweep:

```text
LeRobot 0.4.4 / torch 2.10.0
chunk_size K = 32, n_action_steps = 8
dim_model = 128, heads = 4, feedforward = 512
encoder layers = 2, decoder layers = 1
VAE enabled, latent_dim = 16, VAE encoder layers = 2
dropout = 0.1, KL weight = 1.0
parameters = 1,072,675
AdamW, 2,000 steps, batch 64, learning rate 3e-4, seed 17
```

MPS was unavailable to torch in this run, so the documented CPU fallback was
used. Training took 131.10 seconds. The best sampled validation loss was
0.05017 at step 1900. On validation episodes, first-action L1 was 4.232 mm
(`[5.015, 5.047, 2.633]` mm per axis), and masked chunk L1 was 3.995 mm.
Exact compact metrics are in `artifacts/act_push/training_metrics.json`; the
local checkpoint is `checkpoints/push_act_seed17.pt` and is intentionally Git
ignored.

```bash
python -m learning.train_act_push \
  --dataset data/push_demos_50_seed123.npz \
  --output checkpoints/push_act_seed17.pt \
  --steps 2000 --batch-size 64 --seed 17 --device auto
```

## Runtime and safety

`ACTMotorPolicy.select_action()` is used directly. LeRobot predicts K=32 and
manages its own action queue; eight queued actions are consumed before the next
forward pass. Every selected 3D target then follows the unchanged learned
backend boundary:

```text
finite check
→ 12 cm hard rejection
→ 4 cm maximum Cartesian-step clipping
→ workspace check
→ ManipulationPrimitives.command_cartesian_once
→ CartesianController
→ MuJoCo
```

`ACTPushBackend` imports no expert or `ClassicalPushBackend`, contains no raw
`env.step()`, and has no fallback or task waypoint. The existing `PushSkill`
remains authoritative for displacement, target-region, and on-table success.
AgentRuntime sees only `Push(object, target)`.

## Frozen matched physical result

The same 20 replay-stable initial states used by the previous milestone were
evaluated exactly once:

| Backend | Success | Failure distribution |
|---|---:|---|
| Classical | 20/20 | none |
| One-step BC | 0/20 | 13 timeout, 7 unsafe |
| Simple Chunk BC K=20, H=20 | 2/20 | 6 timeout, 12 unsafe |
| Standard LeRobot ACT K=32, n=8 | **0/20** | 20 timeout |

ACT produced zero unsafe rejections and averaged 685 total control commands,
82 ACT forward calls, and 8.60 cm of EE path. Mean predicted target distance
was 0.884 mm, the mean per-rollout maximum was 18.41 mm, and the global maximum
was 19.83 mm. Cube displacement and target satisfaction were effectively zero
in every trial. Trajectories converged toward approximately
`[-0.26, -0.26, 1.106]`, far from each trial's cube, rather than reaching
contact. Exact trial summaries and one representative sampled timeout trace are
committed under `artifacts/act_push/`; the evaluation command can regenerate
all 20 local traces.

Because nominal success was below 15/20, no robustness probe was run.

## Interpretation and boundary

Standard ACT did not outperform either learned baseline on physical success.
It made the output temporally smooth and eliminated unsafe jumps, showing the
practical contribution of a coherent action queue, but coherence around an
ambiguous hold/waypoint prediction did not recover the missing execution
intent. Its first-action offline error was also worse than the one-step MLP's
2.731 mm, although cross-architecture offline numbers are not task success.

This clean negative result strengthens the interface diagnosis: 89.78% of the
absolute waypoint labels repeat, and near-identical 10D observations occur at
different hidden expert phases. A state-only ACT architecture cannot infer
which phase generated a repeated observation merely by being larger. The
`PushSkill → PushBackend` abstraction itself worked: classical, NumPy BC, chunk
BC, and standard LeRobot ACT remained interchangeable without changing Agent,
planner, semantic verification, or controller safety.

The reproduction is complete and stops here. No follow-up history, phase,
delta-action, temporal ensemble, RNN, diffusion, image, or VLA experiment was
started automatically.
