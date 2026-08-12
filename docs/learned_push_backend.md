# Learned Push Backend Preparation

This exploration keeps the external capability `Push(object, target)` stable
while making its physical realization replaceable:

```text
AgentRuntime
→ SkillExecutor
→ PushSkill
→ PushBackend
   ├── ClassicalPushBackend
   ├── BCPushBackend
   ├── ChunkBCPushBackend
   └── ACTPushBackend
→ ManipulationPrimitives
→ PandaRobot
→ CartesianController
→ robosuite / MuJoCo
```

`PushSkill` still owns semantic preconditions, the two-attempt retry budget,
final displacement/region/tabletop verification, and `SkillResult` creation.
A backend owns one bounded physical attempt and its physical recovery motion.
Planner, validator, AgentRuntime, and SkillExecutor do not know which backend is
selected. `SemanticSkillFactory` accepts a `PushBackend`; the small runtime
factory maps explicit backend configuration to an implementation.

## Trusted control boundary

Learned code never emits a seven-element robosuite action and never calls
`env.step()`. The BC policy predicts an absolute world-frame desired EE xyz.
`BCPushBackend` combines it with the fixed current orientation, rejects NaN and
extreme predictions, enforces workspace and maximum-step limits, and calls
`ManipulationPrimitives.command_cartesian_once`. Only
`CartesianController` converts the resulting pose to an OSC action.

The controller exposes a read-only `CartesianCommandEvent` observer. Recording
does not change a classical trajectory: immediately before each 20 Hz simulator
step it publishes the measured EE pose and trusted absolute pose target.

## State observation contract

Raw `observation.state` has 10 features in this exact order, all in metres and
world coordinates:

```text
[ee_x, ee_y, ee_z,
 cube_x, cube_y, cube_z,
 target_lower_x, target_lower_y,
 target_upper_x, target_upper_y]
```

The raw dataset is not normalized. Training statistics are fitted only from
training episodes and saved in the checkpoint. Orientation is omitted because
the classical v1 trajectory keeps it fixed; gripper state and semantic phase
are deliberately not policy inputs.

## Action contract

`action` has three features:

```text
[desired_ee_x, desired_ee_y, desired_ee_z]
```

It is an absolute desired world-frame EE position in metres for the current
control timestep. Data are recorded at 20 Hz (50 ms). The deterministic backend
wrapper supplies orientation and gripper behavior. Raw actuator commands,
torques, and joint targets are never imitation targets.

LeRobot conversion is direct:

```text
our observations[t] → LeRobot observation.state[t]
our actions[t]      → LeRobot action[t]
episode_offsets     → episode boundaries / indices
metadata_json       → episode task and evaluation metadata
```

## NPZ episode format

The pickle-free NPZ stores `observations`, `actions`, exclusive
`episode_offsets`, `metadata_json`, and `schema_json`. Arrays are concatenated
across episodes while offsets preserve variable episode length. Training data
contain successful expert episodes only; failed expert executions are printed
separately by the collector.

The validated seed-123 dataset contains 50 successful episodes and 32,652
timesteps. Episode lengths are 533 / 653.04 / 1054 (minimum/mean/maximum).
Twenty-six initially sampled configurations were rejected by the pre-execution
settling check; expert execution failures were zero.

Commands:

```bash
python -m datasets.collect_push_demos \
  --episodes 50 --seed 123 --output data/push_demos_50_seed123.npz

python -m datasets.inspect_push_dataset \
  data/push_demos_50_seed123.npz --samples 3 --seed 7
```

## Action chunks

`get_action_chunk(episode, start_t, K)` returns a `(K, 3)` action array and a
boolean `(K,)` validity mask. At episode end, unavailable entries repeat the
last expert action and receive `mask=False`. Therefore:

```text
K = 1  → one-step BC
K > 1  → chunk BC or future ACT targets
```

Future chunk execution must distinguish prediction horizon `K` from execution
horizon `H`:

```text
observe o_t
→ predict [a_t ... a_(t+K-1)]
→ execute H actions, where H <= K
→ re-observe
```

Temporal ensembling was not part of this data-preparation milestone. It was
later enabled only for the standard LeRobot ACT backend, as documented below.

## One-step BC baseline

The baseline is a NumPy MLP:

```text
10 → 64 ReLU → 64 ReLU → 3
```

It uses Adam, an episode-level deterministic 80/20 split, and training-only
normalization. On 40 training and 10 validation episodes, 60 epochs produced:

```text
validation MSE: 1.5478e-5 m²
validation L1:  0.002731 m
per-axis L1:    [0.003123, 0.002462, 0.002608] m
```

Offline error did not translate to full-task success. On 20 replay-stable,
matched initial states:

```text
ClassicalPushBackend: 20/20
BCPushBackend:         0/20
BC failures:           13 POLICY_TIMEOUT, 7 POLICY_ACTION_UNSAFE
```

BC moved the cube by 4.1 mm on average. One rollout moved it 7.8 cm but did not
reach the semantic target. Unsafe predictions were rejected before reaching
the controller.

The main ambiguity is informative: many expert timesteps at the initial state
open the gripper while the Cartesian target remains unchanged. Without time or
phase input, the same observation is paired mainly with a hold command and once
with the command that begins the approach. The one-step conditional mean tends
to remain near the initial pose or drift until a safety limit fires. We do not
add semantic phase input merely to improve this baseline.

## Temporal diagnosis and simple chunk policies

The next milestone implemented two deliberately small extensions without
changing `Push(object, target)` or the trusted controller boundary:

```text
ClassicalPushBackend
        ↓ demonstrations
OneStep BCPushBackend
   ├── ProgressBCPushBackend: [o_t, t/(T-1)] → a_t
   └── ChunkBCPushBackend: o_t → [a_t ... a_(t+K-1)]
```

`ChunkBCPushBackend` explicitly separates prediction horizon K from execution
horizon H. It predicts K absolute Cartesian targets, executes only the first H,
then re-observes. Every target passes through the same finite check, hard
rejection, maximum Cartesian step, workspace check, `Pose`, and
`command_cartesian_once` path as one-step BC. It never receives a primitive,
controller, robot, simulator, or raw action interface.

Dataset diagnosis found that 89.78% of action targets are static under a fixed
3 mm `||a_t-a_(t-1)||` threshold, while only 10.22% change. This is a property
of absolute target commands: an unchanged target can still mean that the OSC
controller is moving toward a fixed waypoint, so it must not be confused with
measured zero EE velocity. The validation moving subset has materially higher
one-step error than the dominant static-target subset.

The measured experiment is in [chunk_bc_experiment.md](chunk_bc_experiment.md).

## Standard LeRobot ACT reproduction

A subsequent milestone added `ACTPushBackend` using the maintained Hugging Face
LeRobot 0.4.4 `ACTPolicy`, rather than a project-local Transformer or CVAE. It
keeps the exact 10D observation and 3D absolute action contract, predicts K=32,
and consumes LeRobot's standard eight-action queue before the next inference.
Every selected target is still handled by the same finite, workspace,
step-limit, primitive, and controller path as the NumPy learned backends.

On the frozen 20 replay-stable initial states it reached 0/20, with 20 timeouts,
zero unsafe actions, and effectively zero cube displacement. This is an honest
negative result: standard ACT made motion smooth but did not recover the hidden
expert phase from the aliased state/absolute-waypoint interface. See
[act_push_backend.md](act_push_backend.md) for the fixed config, training,
physical records, and stopping decision.

### Native ACT Temporal Ensemble

A final inference-only reproduction reused the exact ACT checkpoint and model
weights, set LeRobot's native `temporal_ensemble_coeff=0.01`, and used
`n_action_steps=1` with K=32. Every timestep therefore re-observed the same 10D
state and combined up to 32 overlapping predictions before passing one
absolute target through the unchanged safety and Cartesian-controller path.

On the same frozen 20 states this mode reached 20/20, versus queue ACT's 0/20.
All 20 runs reached the existing 7 cm proximity threshold, moved the cube by
0.208 m on average, and finished with no timeout or unsafe-action rejection.
This demonstrates useful temporal context for this fixed Push distribution; it
does not establish open-scene or task-general learned manipulation. The single
coefficient run completed the requested stopping rule, so no sweep, retraining,
or robustness extension followed.
