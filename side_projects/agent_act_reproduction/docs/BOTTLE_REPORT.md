# S1 Bottle completion report

## 1. Branch and isolation

All work was done on `side/agent-act-reproduction`, created from a clean,
up-to-date `origin/main`. Nothing was merged to `main`. Generated datasets and
checkpoints are local and ignored by Git.

## 2. Side-project structure

```text
side_projects/agent_act_reproduction/
  agent_act_reproduction/
    agent/          MockAgent structured selection
    data/           successful-demo collector and action-chunk dataset
    evaluation/     expert and learned-policy physical trials
    experts/        BottleExpert FSM, collection only
    policies/       pinned LeRobot ACT construction and checkpoint inference
    robot/          bounded Cartesian -> Panda OSC adapter
    runtime/        PolicyRouter and unified command execution
    sim/            Bottle, shelf, table, Panda scene
    tasks/          state observation and physical success checker
    training/       local ACT trainer
  docs/             report and machine-readable measured results
  tests/            agent/schema/router tests
```

## 3. MuJoCo scene

- robosuite Panda on a `0.8 x 0.8 m` table, table top at `z=0.8 m`;
- blue rigid cylinder bottle, radius `0.022 m`, height `0.11 m`;
- fixed raised shelf made from a collision deck plus two supports;
- green visual placement marker on the shelf;
- bottle randomized in x/y by the requested radius and in yaw by `+/-0.10 rad`.

The scene uses real MuJoCo contacts. The shelf and bottle are never teleported
during an episode.

## 4. Panda controller interface

The learned action is absolute Cartesian `[x, y, z, gripper]`. The trusted
adapter bounds XYZ to the declared workspace, computes position error, holds
the Panda's initial top-down quaternion, and sends normalized 7D robosuite
OSC actions `[dx, dy, dz, dRx, dRy, dRz, gripper]`. It does not contain task
phases or waypoints.

## 5-6. Observation and action schemas

| Key | Dim | Fixed order | Units |
|---|---:|---|---|
| `observation.state` | 4 | EE xyz, gripper width | m |
| `observation.environment_state` | 11 | bottle xyz, bottle velocity xyz, shelf target xyz, grasped, progress | m, m/s, bool, fraction |
| `action` | 4 | desired EE xyz, gripper | m, unitless `[-1,1]` |

The exact individual feature names are in `config.py` and embedded in every
dataset file. Training statistics are computed from the 45 training episodes
only and stored inside the local checkpoint.

## 7. ACT source, version, and dependencies

The learned policy class is Hugging Face LeRobot
`lerobot.policies.act.modeling_act.ACTPolicy`, package version `0.4.4`. It is
the maintained implementation derived from the original ACT work, not a local
transformer rewrite. Tested primary dependencies: PyTorch `2.10.0`, torchvision
`0.25.0`, MuJoCo `3.3.7`, robosuite `1.5.2`, NumPy `1.26.4`, einops `0.8.2`.

Training command:

```bash
../../.venv/bin/python -m agent_act_reproduction.training.train \
  --dataset data/bottle_demos_50.npz \
  --output checkpoints/bottle_act.pt \
  --device mps --steps 2000 --batch-size 64 \
  --learning-rate 0.0003 --seed 7
```

## 8-11. Expert and dataset

The collection-only FSM is:

```text
OPEN_GRIPPER -> MOVE_PREGRASP -> DESCEND -> CLOSE_GRIPPER -> LIFT
-> MOVE_ABOVE_SHELF -> PLACE_DESCEND -> OPEN_RELEASE -> RETREAT
-> SETTLE -> DONE
```

- expert gate: `20/20` physical successes (`100%`), target `>=18/20` met;
- expert mean episode length: `226.35` steps;
- demonstrations: `50` successful episodes from `50` attempts;
- dataset timesteps: `11,340`;
- dataset mean/min/max length: `226.8 / 225 / 228`.

Each episode metadata entry contains seed, initial bottle pose, episode length,
physical success, and final bottle pose.

## 12-14. ACT configuration and training

- chunk size `32`, execute `8` actions per query;
- model dim `128`, 4 heads, feed-forward dim `512`;
- 2 transformer encoder layers, 1 decoder layer;
- ACT VAE enabled, latent dim `16`, 2 VAE encoder layers;
- dropout `0.1`, KL weight `1.0`;
- parameter count `1,072,804`;
- split: 45 training episodes / 5 validation episodes;
- device: Apple MPS on M4;
- 2,000 optimizer steps, batch 64, AdamW, LR `3e-4`;
- wall time: `50.61 s`;
- best validation loss: `0.047214` at step 1900;
- final train loss: `0.048860`;
- final validation loss: `0.047578`.

## 15-18. Learned-policy physical evaluation

No evaluation seed was used for training or demonstration collection.

| Distribution | Success | Mean length | Failure distribution |
|---|---:|---:|---|
| nominal, fixed center | `20/20 (100%)` | `162.05` | none |
| x/y `+/-1 cm` | `20/20 (100%)` | `161.65` | none |
| OOD x/y `+/-3 cm` | `11/20 (55%)` | `232.75` | 9 outside shelf and wrong height |

The dominant OOD failure is missing the initial grasp when the bottle begins
beyond the training envelope. The bottle remains on or is nudged across the
table, so final checks report both wrong height and outside shelf. The OOD probe
was not used to tune or retrain the policy; it exposes the intended capability
boundary.

## 19-20. MockAgent and PolicyRouter

Supported input produces:

```json
{"status":"OK","request":{"skill":"place_bottle_on_shelf","args":{}},"available_capabilities":["place_bottle_on_shelf"]}
```

Unsupported input such as `Draw a circle.` produces:

```json
{"status":"CANNOT_EXECUTE","request":null,"available_capabilities":["place_bottle_on_shelf"]}
```

`PolicyRouter` maps the one skill to `checkpoints/bottle_act.pt`. Replacing
`MockAgent` with an LLM requires only another object returning the same
`AgentResponse` / `SkillRequest` boundary.

## 21. Complete trace

Representative nominal learned rollout (`seed=4000`):

```text
USER
Put the bottle on the shelf.

AGENT
selected_skill = place_bottle_on_shelf

POLICY
implementation = Hugging Face LeRobot ACTPolicy 0.4.4
checkpoint = checkpoints/bottle_act.pt

EXECUTION
step 1:   bottle=[0.05001,-0.09999,0.85579], grasped=false
step 75:  bottle=[0.04939,-0.10029,0.85482], grasped=true
step 125: bottle=[0.10453,0.07680,1.06437], grasped=true
step 150: bottle=[0.10501,0.20753,0.96830], grasped=false
step 162: final bottle=[0.10423,0.20860,0.96794]

RESULT
SUCCESS
```

## 22. Hidden assistance audit

There is no hidden scripted assistance. Runtime has no import of the expert
package, no task-specific Cartesian path, no classical task FSM, no teleport,
and no expert fallback after failure. ACT produces the complete motor sequence.
Only workspace clipping, fixed-orientation holding, physical timeout, and the
success checker remain outside the learned policy.

## 23-24. Git

The final commit hash and push status are reported after the commit and remote
push complete. This branch must not be merged to `main` as part of this work.

## 25. Engineering assessment

For this fixed-workspace hackathon problem, the Agent layer is almost trivial;
the real work is simulator contact tuning, demonstration reliability, strict
feature ordering, normalization, and making the learned rollout physically
auditable. Once the deterministic expert was reliable, 50 demonstrations and a
roughly one-million-parameter state ACT trained in under a minute on MPS and
executed the nominal task robustly. The sharp drop from 100% at `+/-1 cm` to 55%
at `+/-3 cm` is the clearest boundary: this is a learned fixed-distribution
motor routine selected by language, not a general manipulation system.
