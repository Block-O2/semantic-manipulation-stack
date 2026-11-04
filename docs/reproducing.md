# Reproducing and inspecting the stack

This guide separates runnable classical demos, inspection of saved evidence,
and learned-policy experiments that require locally generated data and weights.
The 2026-11-05 validation is described in [evidence_audit.md](evidence_audit.md).

## Install and test

From the repository root, use Python 3.11 and the README virtual-environment
commands. `pyproject.toml` allows Python 3.10–3.12; this audit tested only 3.11
on macOS ARM64. A fresh base environment needs no LLM service or checkpoint:

```bash
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest -q
```

The core runtime imports learned backends lazily. ACT/Diffusion tests use
synthetic temporary checkpoints, not the historical trained weights. Their
test modules skip if their optional dependencies are absent. The full 103-test
run used an existing environment; it is not evidence of a
clean full-extra installation. See the dependency status below before trying
ACT training or tests.

`act` pins LeRobot 0.4.4 / torch 2.10.0 / torchvision 0.25.0 / einops 0.8.2;
`diffusion` pins torch 2.10.0 / diffusers 0.35.2. They are optional, substantial
downloads. `analysis` adds SciPy for dataset diagnostics. The `datasets/`
package name is historical: the narrow LeRobot import shim in
`learning/lerobot_compat.py` handles its collision with Hugging Face `datasets`.

Base constraints are robosuite 1.5.2, MuJoCo 3.3.x, and NumPy 1.x. A newer
MuJoCo removes `MjData.qM`, which this robosuite version accesses; do not upgrade
it without a compatibility check. The package has ranges rather than a full
transitive lock, so an installation is not a bit-identical environment.

robosuite may warn about private macros, optional robot models, or GR1 IK.
The Panda/BASIC OSC path used here does not require those optional components;
the demos and tests completed with those warnings. No private macro file is
needed for these commands.

## Optional learning dependency status

The audit's fresh `pip install --dry-run -e '.[dev,act,diffusion]'` did not
resolve successfully. LeRobot 0.4.4 requires `rerun-sdk>=0.24,<0.27`, whose
NumPy requirement conflicts with this project's `numpy<2`. The historical
installed environment runs all 103 tests, but `pip check` reports
`rerun-sdk 0.26.2` requires `numpy>=2` while NumPy is 1.26.4. It is **not** a
clean dependency-consistent reference environment. Do not infer an installable
ACT environment from its passing tests.

The ACT extra remains recorded in `pyproject.toml` as the historical intent;
its `pip install -e '.[dev,act]'` command is currently blocked under normal
resolution. A compatible upstream/dependency arrangement is still needed.
This audit does not bypass dependencies with `--no-deps`, upgrade the simulation
to NumPy 2, or change the ACT algorithm to hide this limitation. The ACT commands
below are historical training/runtime recipes conditional on a working local
ACT environment, not a verified fresh setup. Diffusion is checked separately
because it does not need the LeRobot package. A fresh
`python -m pip install -e '.[dev,diffusion]'` completed, `pip check` passed, and
`python -m pytest -q` passed **95 tests with 1 ACT module skipped**. Base-only
installation passed **89 tests with 2 optional modules skipped**. The full
103-test result remains limited to the existing ACT environment described above.

## Headless and GUI demos

```bash
python -m demos.semantic_agent --no-render --inspect-seconds 0
python -m demos.push_test --no-render --inspect-seconds 0
python -m demos.semantic_agent --no-render --inspect-seconds 0 --disturbance object-lost
python -m demos.dynamic_scenarios --scenario all
```

The scripted scenarios assert seven expected outcomes, including successful
rearrangement/replanning and clean refusal of unreachable/unsupported goals.
Exit 0 for the set does not mean every individual goal succeeded.

On macOS, the interactive MuJoCo viewer must use `mjpython` from the same
virtual environment, not plain Python:

```bash
mjpython -m demos.semantic_agent --inspect-seconds 2
mjpython -m demos.push_test --inspect-seconds 2
mjpython -m demos.dynamic_playground
```

`demos.dynamic_playground --no-render` still pauses at each skill for terminal
commands. Enter `continue` twice for nominal Pick/Place. For unattended nominal
execution use `demos.semantic_agent`; for unattended disturbances use
`demos.dynamic_scenarios`. See [dynamic_playground.md](dynamic_playground.md).

The default planner is deterministic. `--planner llm` additionally needs an
accessible OpenAI-compatible service and `SEMANTIC_AGENT_MODEL`,
`SEMANTIC_AGENT_API_KEY`, and optionally `SEMANTIC_AGENT_BASE_URL`. It was not
called in this audit and is not part of the reported success rates.

## Nominal evaluations

```bash
python -m evaluation.pick_place_trials --trials 20 --seed 47
python -m evaluation.agent_trials --trials 20 --seed 59
python -m evaluation.push_trials --trials 20 --seed 127
```

All three reached 20/20 in the audit. Pick/Place and Agent randomize all three
cube positions in the configured fixed scene (`sim/scene.py`: x ∈ [-0.18,0.18],
y ∈ [-0.18,0.04], minimum cube separation 0.075 m). They soft-reset one
environment across trials. Push uses a fresh simulator per trial, moves the
robot to a standby pose, removes two distractor cubes explicitly, and samples
x ∈ [-0.10,0.08], y ∈ [-0.15,0.12]. Before execution it rejects placements that
move more than 3 mm during ten settling steps or leave the table. This rerun
had **14 placement rejections** for 20 accepted trials; one accepted trial used
a local retry (mean attempts 1.05). Nominal Pick/Place used no local retries;
nominal Agent used no replans. These protocols are not the matched learned
comparison or a broad robustness study.

Per-trial stdout records and summaries are saved in
[`artifacts/portfolio_validation/nominal_trials.json`](../artifacts/portfolio_validation/nominal_trials.json).
The evaluators' exit threshold is 90%; inspect the reported counts rather than
interpreting exit 0 as proof of 20/20.

## Inspect historical learned evidence without weights

```bash
# Standard library only; use the script path to avoid evaluation/__init__ imports.
python evaluation/audit_saved_results.py
```

It recounts success/failure and target flags, checks trial IDs and the exact
initial positions/seeds across the four committed comparison files, and checks
episode-split metadata. It does not execute a learned policy. The records cover
20 replay-stable demonstration states: **17 train, 3 validation**. Four
candidates among indices 0–23 were skipped as replay-unstable. This is a
development-set diagnostic, not held-out task/scene generalization.

## Learned Push is not a one-command demo

No model weights or datasets are tracked in Git, and no public checkpoint
download is provided. Existing result records document a historical local run.
A new reader must collect data, train a model, then evaluate it. Retraining is
a new experiment and need not recover the historical checkpoint hash or score.

The historical dataset used 50 **successful** classical episodes, 32,652 steps
at 20 Hz, seed 123. Failed expert executions are excluded; initial unstable
placements are rejected. Inputs are 10D `[ee_xyz, cube_xyz, region_lower_xy,
region_upper_xy]`; outputs are 3D absolute desired EE positions in metres.
The seed-17 split is 40 train / 10 validation episodes, with train-only
normalization. The raw historical NPZ and its hash are not in the public
repository, so its exact contents cannot be independently checked from Git.

```bash
python -m datasets.collect_push_demos \
  --episodes 50 --seed 123 --output data/push_demos_50_seed123.npz

# ACT: conditional recipe for an existing runtime, not a fresh-install command.
# The act extra currently has the NumPy resolver conflict documented above.
python -m learning.train_act_push \
  --dataset data/push_demos_50_seed123.npz \
  --output checkpoints/push_act_seed17.pt \
  --steps 2000 --batch-size 64 --seed 17 --device cpu
python -m demos.push_test --no-render --inspect-seconds 0 \
  --push-backend act --act-execution-mode temporal_ensemble \
  --checkpoint checkpoints/push_act_seed17.pt --device cpu

# Diffusion: one 10,000-step configuration (historical CPU training ~2.44 h).
python -m pip install -e '.[dev,diffusion]'
python -m learning.train_diffusion_push \
  --dataset data/push_demos_50_seed123.npz \
  --output checkpoints/push_diffusion_seed17.pt \
  --steps 10000 --batch-size 256 --learning-rate 1e-4 --seed 17 --device cpu
python -m demos.push_test --no-render --inspect-seconds 0 \
  --push-backend diffusion --checkpoint checkpoints/push_diffusion_seed17.pt --device cpu
```

To evaluate the fixed indices, use new output paths to retain historical JSON:

```bash
python -m evaluation.act_temporal_ensemble_comparison --device cpu \
  --dataset data/push_demos_50_seed123.npz --checkpoint checkpoints/push_act_seed17.pt \
  --output /tmp/act-ensemble.json --trajectory-output /tmp/act-ensemble-trace.json
python -m evaluation.diffusion_push_comparison --device cpu \
  --dataset data/push_demos_50_seed123.npz --checkpoint checkpoints/push_diffusion_seed17.pt \
  --output /tmp/diffusion.json --trajectory-output /tmp/diffusion-trace.json
```

The indices assume the original episode ordering. If recollection changes
rejections or episode order, those indices no longer identify the historical
states. Default evaluator paths overwrite committed reports, so use explicit
new output paths. Queue ACT is retained as a negative control and is the CLI
default unless `--act-execution-mode temporal_ensemble` is supplied.

The queue/ensemble same-weight comparison is documented in
[act_push_backend.md](act_push_backend.md). Historical ensemble hash:
`9b60ed6880a151d5805459ef14e041453aba686ccf7cdc5ddf8631a2ce337ab4`;
Diffusion hash: `4b84219ec59254cbe693b2ad838dbc1c548969554ffb34a50faaae58e7b0049e`.
These identify local artifacts; they are not download links. Queue JSON itself
has no checkpoint hash, so the public record cannot independently prove the
same-weight provenance beyond the source/report. Full training and the frozen
learned physical benchmarks were not rerun in this portfolio audit.

## Record the demo

```bash
python -m demos.record_portfolio --output /tmp/semantic-portfolio.gif
```

Uses MuJoCo Renderer plus Pillow, already supplied through core robosuite
requirements; no video encoder or new dependency is needed. On macOS the
process needs access to the graphics service even though it opens no GUI.
An isolated process without that access failed with `invalid CoreGraphics
connection`; recording succeeded with normal local graphics access. Truly
headless Linux rendering may need EGL/OSMesa and was not tested here.

The committed [GIF](media/semantic-stack.gif) and [JSON](media/semantic-stack.json)
were captured from the existing classical/Agent execution at source algorithm
commit `1002984923a14ae8c1b466b6845827d3a53a59cd`, using seed 67 per clip.
The JSON records package versions, capture-script hash, media hash, complete
AgentResult, semantic residuals, and final state. The recorder is an observer;
it adds captions and samples every eighth 20 Hz control command for 4× playback
at 10 fps, plus title/outcome holds. Frames are scaled to 560×388 for README delivery. It does not change control algorithms.
The object-loss clip calls the existing `ObjectLostAfterPick` hook: it explicitly
moves the cube to the table and the Agent replans once. This shows semantic
recovery, not a skill-local grasp retry, learned recovery, or natural fault rate.
