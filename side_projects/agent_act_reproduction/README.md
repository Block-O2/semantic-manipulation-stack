# Agent + ACT three-demo reproduction

This side project is a deliberately bounded reproduction of a hackathon-style
robotics stack:

```text
natural-language command
  -> MockAgent chooses one predefined skill
  -> PolicyRouter selects one task-specific ACT checkpoint
  -> bounded Cartesian Panda adapter
  -> robosuite / MuJoCo physics
  -> task-specific physical success checker
```

It implements three independent motor skills:

| Skill id | Command example | Simplified simulation task |
|---|---|---|
| `place_bottle_on_shelf` | 把瓶子放到架子上 | Grasp, transfer, release, and stabilize a bottle in a shelf target |
| `pull_tissue_from_box` | 从纸巾盒抽出一张纸 | Grasp and pull a thin rigid sheet from a narrow slot |
| `draw_horizontal_line` | 拿笔在纸上画一横 | Grasp a square pen holder, make tip contact, draw, and lift |

This is not one general robot policy. Each task has its own demonstrations,
checkpoint, observation schema, expert gate, and physical success condition.
Unknown commands return `CANNOT_EXECUTE`.

## Chinese explainer page

[`web/index.html`](web/index.html) is the final explainer. It is one standalone
HTML file with inline CSS, inline JavaScript, and three Base64-embedded ACT-only
GIFs. Double-click it to open it offline; no build step or network connection is
required. A local server also works:

```bash
cd side_projects/agent_act_reproduction/web
python3 -m http.server 8000
```

Then visit `http://localhost:8000/`.

To refresh the embedded demos, first record successful learned-policy rollouts,
then run the embedding helper. It replaces the three existing GIFs in Bottle,
Tissue, Draw order (and also accepts the original three named placeholders):

```bash
../../.venv/bin/python -m agent_act_reproduction.record_gifs bottle \
  --checkpoint checkpoints/bottle_act.pt --output outputs/bottle.gif --seed 4000
../../.venv/bin/python -m agent_act_reproduction.record_gifs tissue \
  --checkpoint checkpoints/tissue_act.pt --output outputs/tissue.gif --seed 5000
../../.venv/bin/python -m agent_act_reproduction.record_gifs draw \
  --checkpoint checkpoints/draw_act.pt --output outputs/draw.gif --seed 5100
../../.venv/bin/python -m agent_act_reproduction.embed_web_gifs web/index.html \
  --bottle outputs/bottle.gif --tissue outputs/tissue.gif --draw outputs/draw.gif
```

## Working directory and environment

Run commands from this directory, not the repository root:

```bash
cd side_projects/agent_act_reproduction
```

The verified environment is Python 3.11 on Apple M4 with LeRobot `0.4.4`,
PyTorch `2.10.0`, MuJoCo `3.3.7`, robosuite `1.5.2`, and NumPy `1.26.4`.
LeRobot supplies `ACTPolicy`; this project owns the task environments, dataset
wrapper, training loop, checkpoint metadata, runtime routing, and evaluation.
Datasets and checkpoints remain local and are ignored by Git.

## Reproduce Tissue or Draw

Use `tissue` or `draw` consistently below:

```bash
# 1. Expert reliability gate
../../.venv/bin/python -m agent_act_reproduction.evaluation.tasks tissue \
  --mode expert --trials 20 --seed 1000 --perturbation-m 0.006

# 2. Collect exactly 50 successful demonstrations
../../.venv/bin/python -m agent_act_reproduction.data.collect_tasks tissue \
  --output data/tissue_demos_50.npz --episodes 50 \
  --seed 2000 --perturbation-m 0.006

# 3. Train a task-specific ACT checkpoint
../../.venv/bin/python -m agent_act_reproduction.training.train \
  --task-name tissue --dataset data/tissue_demos_50.npz \
  --output checkpoints/tissue_act.pt --device mps \
  --steps 2000 --batch-size 64 --learning-rate 0.0003 --seed 7

# 4. Run the full language -> routing -> ACT -> robot chain
../../.venv/bin/python -m agent_act_reproduction.runtime.execute_tasks \
  '从纸巾盒抽出一张纸' --device cpu

# 5. Closed-loop ACT evaluation
../../.venv/bin/python -m agent_act_reproduction.evaluation.tasks tissue \
  --mode act --checkpoint checkpoints/tissue_act.pt \
  --trials 20 --seed 5000 --perturbation-m 0.006 --device cpu
```

Bottle retains its original task-specific collection and evaluation entry
points documented in [`docs/BOTTLE_REPORT.md`](docs/BOTTLE_REPORT.md).

## No hidden expert assistance

Experts are imported only by demonstration collection and expert evaluation.
The learned runtime imports no expert package, phase state, task waypoint,
teleport, scripted fallback, or retry recovery. ACT outputs absolute Cartesian
XYZ targets plus a gripper command. `PandaCartesianAdapter` only clips those
targets to the workspace, holds the fixed top-down orientation, and passes them
to robosuite's OSC controller.

Success is checked from simulation state and must remain true for eight
consecutive physics steps. In Draw, the visible line is created only from
actual `pen_tip_geom` to paper contact; there is no pre-drawn result.

## Measured result snapshot

| Task | Expert gate | Demonstrations | Nominal ACT | Small perturbation ACT |
|---|---:|---:|---:|---:|
| Bottle | 20/20 at ±1 cm | 50 / 11,340 steps | 20/20 | 20/20 at ±1 cm; 11/20 at ±3 cm |
| Tissue | 20/20 at ±6 mm | 50 / 6,574 steps | 20/20 | 20/20 at ±6 mm |
| Draw | 20/20 at ±6 mm | 50 / 9,685 steps | 20/20 | 11/20 at ±6 mm |

These are fixed-scene simulation results, not evidence of open-world
generalization or real-robot robustness. Machine-readable details are in
[`docs/results.json`](docs/results.json).

## Tests

```bash
../../.venv/bin/python -m pytest -q tests
```
