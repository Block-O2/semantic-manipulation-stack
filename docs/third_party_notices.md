# Third-party notices

## Repository and contribution boundary

There is no repository-wide LICENSE for the author's original code. This
portfolio update does not assign one or infer permission from dependency
licenses. The existing MIT notice for the adapted Diffusion code remains below.

The project's own work is the scene/runtime integration, symbolic contracts,
validation, bounded skills, outcome checks, Push adapters, demonstration/data
pipeline, and diagnostic experiments. It does not claim authorship of MuJoCo,
robosuite, ACT, Temporal Ensemble, or the Diffusion Policy algorithm.

| Upstream | Use here | Upstream license reference |
|---|---|---|
| MuJoCo 3.3.7 | Physics and rendering via installed package | [Apache-2.0](https://github.com/google-deepmind/mujoco/blob/3.3.7/LICENSE) |
| robosuite 1.5.2 | Panda/tabletop assets, environment base, OSC | [MIT, with MuJoCo notice](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.2/LICENSE) |
| Hugging Face LeRobot 0.4.4 | Imported ACTPolicy and native Temporal Ensemble | [Apache-2.0 and included third-party notices](https://github.com/huggingface/lerobot/blob/v0.4.4/LICENSE) |
| Stanford Diffusion Policy | Adapted low-dimensional conditional U-Net layout | [MIT at source commit](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/LICENSE) |
| Hugging Face diffusers 0.35.2 | Imported DDPMScheduler | [Apache-2.0](https://github.com/huggingface/diffusers/blob/v0.35.2/LICENSE) |

These are direct integration boundaries, not a complete transitive dependency
license inventory. Package/asset-specific notices remain applicable. The
portfolio GIF renders this project's scene using the installed robosuite Panda,
table, floor, and visual assets; no external stock media is used, and no
third-party mesh/package source or trained checkpoint is added by this update.

## real-stanford/diffusion_policy

`learning/diffusion_push.py` minimally adapts the conditional 1D U-Net and
low-dimensional Diffusion Policy layout from commit
`5ba07ac6661db573af695b419a7947ecb704690f` of
`real-stanford/diffusion_policy`.

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
