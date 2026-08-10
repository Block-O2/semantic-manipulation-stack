"""Record successful ACT-only rollouts as local GIF source assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from agent_act_reproduction.runtime.execute_tasks import execute_command


COMMANDS = {
    "bottle": "把瓶子放到架子上",
    "tissue": "从纸巾盒抽出一张纸巾",
    "draw": "拿笔在纸上画一横",
}
SKILLS = {
    "bottle": "place_bottle_on_shelf",
    "tissue": "pull_tissue_from_box",
    "draw": "draw_horizontal_line",
}


def record(*, task: str, checkpoint: Path, output: Path, seed: int, device: str) -> dict[str, object]:
    frames: list[Image.Image] = []

    def capture(env, step: int) -> None:
        if step % 2:
            return
        pixels = env.sim.render(width=640, height=480, camera_name="frontview")
        pixels = np.flipud(np.asarray(pixels, dtype=np.uint8))
        frames.append(Image.fromarray(pixels).crop((0, 60, 640, 420)))

    result = execute_command(
        command=COMMANDS[task],
        checkpoints={SKILLS[task]: checkpoint},
        seed=seed,
        perturbation_m=0.0,
        device=device,
        offscreen=True,
        frame_callback=capture,
        verbose=False,
    )
    if not result.get("success"):
        raise RuntimeError(f"ACT rollout failed; refusing to record as demo: {result}")
    if len(frames) < 2:
        raise RuntimeError("Renderer returned too few frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    master_palette = frames[len(frames) // 2].convert(
        "P", palette=Image.Palette.ADAPTIVE, colors=64
    )
    quantized = [
        frame.quantize(palette=master_palette, dither=Image.Dither.NONE)
        for frame in frames
    ]
    quantized[0].save(
        output,
        save_all=True,
        append_images=quantized[1:],
        duration=100,
        loop=0,
        optimize=True,
        disposal=1,
    )
    summary = {
        "task": task,
        "seed": seed,
        "success": True,
        "episode_length": result["episode_length"],
        "frames": len(frames),
        "fps": 10,
        "resolution": "640x360",
        "bytes": output.stat().st_size,
        "path": str(output.resolve()),
        "hidden_scripted_assistance": result["hidden_scripted_assistance"],
    }
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("bottle", "tissue", "draw"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    args = parser.parse_args()
    record(task=args.task, checkpoint=args.checkpoint, output=args.output, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
