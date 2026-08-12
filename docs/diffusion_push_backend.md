# State-Based Diffusion Policy Push Backend

M7.6 asks whether a second mature sequence-policy paradigm can drive the same
physical Push skill without changing its task, safety, controller, or semantic
success boundary:

```text
50 existing Classical Push demonstrations
→ state-based conditional Diffusion Policy
→ DiffusionPushBackend
→ existing PushSkill and trusted Cartesian controller
→ the frozen 20-state MuJoCo evaluation
```

No image, expert phase, progress scalar, previous action, scripted waypoint,
fallback, teleport, or raw simulator action is supplied to the policy.

## Implementation source and fixed configuration

The algorithm and defaults follow the official
[`real-stanford/diffusion_policy`](https://github.com/real-stanford/diffusion_policy)
low-dimensional U-Net configuration at commit
`5ba07ac6661db573af695b419a7947ecb704690f`. The conditional residual 1D U-Net
is minimally adapted under its MIT license to this repository's 10D state and
3D action contract. Hugging Face `diffusers==0.35.2` supplies the maintained
`DDPMScheduler`; `torch==2.10.0` supplies training and inference. The complete
license is retained in [third_party_notices.md](third_party_notices.md).

One configuration was selected before training and was not swept:

```text
observation horizon To = 2
prediction horizon Tp = 16
execution horizon Ta = 8
global observation conditioning
1D conditional U-Net channels = [64, 128, 256]
diffusion timestep embedding = 128
kernel = 5, GroupNorm groups = 8, FiLM scale and bias = enabled
parameters = 4,488,323
DDPM train timesteps = 100, inference denoising steps = 100
beta schedule = squaredcos_cap_v2, prediction target = epsilon
```

The official low-dimensional model uses the same `To=2 / Tp=16 / Ta=8`,
global conditioning, U-Net form, scheduler, and 100-step epsilon objective.
Only the channel widths and timestep embedding are reduced from
`[256, 512, 1024] / 256` to keep this small 10D/3D task practical on CPU. This
is an implementation-size adaptation, not an architecture or scheduler study.

## Forward noising and reverse denoising

Each training window contains two observations and a clean 16-step absolute
Cartesian action trajectory `x_0`. Training samples a diffusion timestep and
Gaussian noise, forms `x_k` with the standard DDPM forward process, then asks
the conditional U-Net to predict that noise. MSE is applied only to valid
action positions.

At runtime, sampling starts from a Gaussian `(16, 3)` action trajectory. One
hundred reverse DDPM steps condition it on the two normalized observations.
The output is inverse-normalized to metres, actions 1 through 8 are executed,
and every realized control observation is appended before the next sample.
The initial history repeats the first observation; action windows repeat the
nearest endpoint for storage, carry a validity mask, and never cross episode
boundaries.

## Data and training contract

The unchanged NPZ contains 50 successful episodes, 32,652 aligned 20 Hz
timesteps, and these features:

```text
observation = [ee_xyz, cube_xyz, target_lower_xy, target_upper_xy]  # 10D, m
action      = absolute desired_ee_xyz                               # 3D, m
```

Seed 17 fixes the episode-level split at 40 train / 10 validation episodes.
Per-dimension min/max statistics are fit only on training episodes and map
observations and actions to `[-1, 1]`; the checkpoint stores min, max, mean,
standard deviation, scale, and offset. Runtime inverse normalization restores
absolute targets in metres.

The single training command is:

```bash
python -m pip install -e '.[dev,diffusion]'
python -m learning.train_diffusion_push \
  --dataset data/push_demos_50_seed123.npz \
  --output checkpoints/push_diffusion_seed17.pt \
  --steps 10000 --batch-size 256 --learning-rate 1e-4 \
  --seed 17 --device auto
```

It uses AdamW with betas `(0.95, 0.999)`, weight decay `1e-6`, a 500-step
warmup followed by cosine decay, EMA power `0.75`, and local CPU fallback when
MPS is unavailable. Dataset and checkpoint files remain Git ignored; compact
metrics and physical traces are committed under `artifacts/diffusion_push/`.

## Runtime and trusted boundary

`DiffusionPushBackend` implements the existing `PushRequest +
PushExecutionContext → PushBackendResult` protocol. Each sampled absolute xyz
target passes through the unchanged learned-backend checks:

```text
inverse normalization
→ finite check
→ 12 cm hard rejection
→ 4 cm maximum Cartesian-step clipping
→ workspace check
→ ManipulationPrimitives.command_cartesian_once
→ trusted Cartesian controller
→ MuJoCo
```

The policy neither calls an expert nor imports `ClassicalPushBackend`, and it
has no `env.step()` access or fallback. `PushSkill`, not the neural policy,
remains responsible for cube displacement, target-region, and on-table success.

## Training and offline result

MPS was unavailable, so the single run used CPU. Training took 8,795.36
seconds for 10,000 optimizer steps (batch 256, about 98.9 passes over the
25,876 train windows). Best validation diffusion loss was `0.0045409` at step
10,000. The one prescribed sampling pass covered 2,048 validation windows:

```text
sampled full-trajectory action L1 = 2.246 mm
first executable action L1       = 2.168 mm
per-axis first-action L1         = [2.286, 2.107, 2.112] mm
horizon-wise L1 range            = 1.939 to 2.598 mm
```

The 100-step sanity trace starts from Gaussian action noise. Mean absolute
change between consecutive physical samples was 172.46 mm at reverse
iteration 2 and fell to 3.75 mm at iteration 100. The statistic is not a task
metric, but confirms that the configured reverse process updates and settles
the sampled trajectory. Exact values are in
`artifacts/diffusion_push/training_metrics.json`.

## Frozen physical result

The same 20 replay-stable states were evaluated exactly once. The checkpoint
hash stayed unchanged at
`4b84219ec59254cbe693b2ad838dbc1c548969554ffb34a50faaae58e7b0049e`.

| Metric | Diffusion Policy |
|---|---:|
| Success | **20/20** |
| Target satisfied | 20/20 |
| Reached 7 cm EE-cube proximity | 20/20 |
| Timeout / unsafe | 0 / 0 |
| Mean control steps | 579.25 |
| Mean cube displacement | 0.2074 m |
| Mean / global minimum EE-cube distance | 0.05298 / 0.05148 m |
| Mean EE path | 0.7244 m |
| Mean sampling calls | 68.4 |
| Mean predicted / executed target step | 6.484 / 6.484 mm |
| CPU sampling latency, mean / p95 | 244.4 / 284.7 ms |

All trials succeeded on the first attempt. Predicted and executed step means
were identical: the sampled trajectories stayed inside the 4 cm clipping
limit rather than relying on clipping to become executable. The 100-step
sampler is much slower than the environment's 50 ms control period, but
executing eight actions per sample reduces the number of calls. The prior ACT
evaluation did not record a matched inference-latency metric, so this milestone
does not claim a measured ACT-to-Diffusion speed ratio.

## ACT comparison and boundary

| Policy | Temporal mechanism | Frozen result |
|---|---|---:|
| ACT queue | predict K=32, execute 8 queued actions | 0/20 |
| ACT Temporal Ensemble | infer every step, overlap up to 32 chunks | 20/20 |
| Diffusion Policy | condition on two states, denoise Tp=16, execute Ta=8 | 20/20 |

ACT Temporal Ensemble preserves intent by combining overlapping predictions
from many earlier observations. Diffusion instead samples one coherent future
trajectory from a short observation history and commits to eight actions
before replanning. On this frozen task, the full Diffusion formulation supplied
enough temporal context for successful Push. Because no `To=1` ablation was
allowed, the result cannot attribute success to observation history alone or
separate it from trajectory denoising and receding-horizon execution.

The main architectural result is that classical FSM, ACT, and Diffusion Policy
all remain interchangeable behind `PushBackend`; Agent, planner, PushSkill,
semantic success, and trusted control did not change. Two distinct mature
sequence-policy paradigms now solve the fixed reproduction, so another learned
policy would add little evidence without a new research question. This result
is interpreted only for the fixed-workspace, state-only Push distribution; it
is not evidence of perception, new-task learning, open-scene generalization,
real-robot robustness, or general manipulation intelligence.
