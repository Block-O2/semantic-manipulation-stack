# Agent + ACT Bottle Reproduction

This directory is an intentionally small, state-only reproduction of the
hackathon architecture:

```text
natural-language command
  -> MockAgent structured skill request
  -> PolicyRouter
  -> task-specific Hugging Face LeRobot ACT checkpoint
  -> bounded Cartesian Panda adapter
  -> robosuite / MuJoCo physics
  -> BottleSuccessChecker
```

Only `place_bottle_on_shelf` exists in S1. Tissue and Draw are deliberately
not implemented. Unknown commands return `CANNOT_EXECUTE` and list the one
available capability.

## Important working-directory note

Run commands from this directory, not the repository root:

```bash
cd side_projects/agent_act_reproduction
```

The parent repository already contains a top-level Python package named
`datasets`. Starting here prevents that package from shadowing Hugging Face's
third-party `datasets` package when LeRobot imports.

## Environment

The verified environment is Python 3.11 on Apple M4 with:

- Hugging Face LeRobot `0.4.4` (`ACTPolicy`)
- PyTorch `2.10.0`, torchvision `0.25.0`
- MuJoCo `3.3.7`, robosuite `1.5.2`
- NumPy `1.26.4`, einops `0.8.2`

LeRobot is used as the mature ACT implementation. The side project owns only
the local episode-aware dataset wrapper, normalization, training loop, and
checkpoint metadata. Nothing is uploaded to Hugging Face Hub.

## Reproduce S1 Bottle

Assuming the parent repository's `.venv` is active or using the explicit
relative interpreter below:

```bash
# 1. Expert reliability gate
../../.venv/bin/python -m agent_act_reproduction.evaluation.expert \
  --trials 20 --seed 1000 --perturbation-m 0.01

# 2. Collect exactly 50 successful demonstrations (ignored by Git)
../../.venv/bin/python -m agent_act_reproduction.data.collect \
  --output data/bottle_demos_50.npz \
  --episodes 50 --seed 2000 --perturbation-m 0.01

# 3. Train local ACT checkpoint (ignored by Git)
../../.venv/bin/python -m agent_act_reproduction.training.train \
  --dataset data/bottle_demos_50.npz \
  --output checkpoints/bottle_act.pt \
  --device mps --steps 2000 --batch-size 64 \
  --learning-rate 0.0003 --seed 7

# 4. Complete command -> Agent -> ACT -> robot trace
../../.venv/bin/python -m agent_act_reproduction.runtime.execute \
  'Put the bottle on the shelf.' \
  --checkpoint checkpoints/bottle_act.pt --device mps

# 5. Physical evaluations
../../.venv/bin/python -m agent_act_reproduction.evaluation.policy \
  --checkpoint checkpoints/bottle_act.pt --trials 20 \
  --seed 5000 --perturbation-m 0.0 --device cpu --label nominal
../../.venv/bin/python -m agent_act_reproduction.evaluation.policy \
  --checkpoint checkpoints/bottle_act.pt --trials 20 \
  --seed 6000 --perturbation-m 0.01 --device cpu --label small_perturbation
../../.venv/bin/python -m agent_act_reproduction.evaluation.policy \
  --checkpoint checkpoints/bottle_act.pt --trials 20 \
  --seed 7000 --perturbation-m 0.03 --device cpu --label ood_3cm

# Tests
../../.venv/bin/python -m pytest -q tests
```

On macOS, rendered rollouts must be run with robosuite's `mjpython` equivalent
and `--render`; all reported metrics were headless physical simulations.

## Explicit observation schema

All model inputs are `float32`. Feature order is fixed in
`agent_act_reproduction/config.py` and copied into the dataset metadata.

| ACT key | Dim | Ordered features | Units |
|---|---:|---|---|
| `observation.state` | 4 | EE x/y/z; gripper width | m |
| `observation.environment_state` | 11 | bottle x/y/z; bottle vx/vy/vz; shelf target x/y/z; grasped; episode progress | m, m/s, bool, fraction |
| `action` | 4 | absolute desired EE x/y/z; gripper command | m, scalar in `[-1, 1]` |

The policy never writes MuJoCo state or torque. `PandaCartesianAdapter` clips
the learned absolute Cartesian target to a safe workspace, holds the fixed
top-down orientation, and converts the command to robosuite's Panda OSC action.

## No hidden expert assistance

`BottleExpert` is imported by collection and expert evaluation only. The
runtime module does not import the expert package and contains no Bottle FSM,
task waypoint, teleport, policy fallback, or retry recovery. During rollout,
the learned ACT checkpoint is responsible for open, approach, descend, grasp,
lift, transfer, place, release, and retreat. The adapter provides only clipping,
orientation holding, and the standard OSC interface.

See [docs/BOTTLE_REPORT.md](docs/BOTTLE_REPORT.md) for measured results and the
engineering assessment.
