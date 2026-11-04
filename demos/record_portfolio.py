"""Record real classical/semantic MuJoCo runs as a captioned GIF (no weights)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import mujoco
from PIL import Image, ImageDraw

from evaluation.disturbances import ObjectLostAfterPick
from planner import Goal, PlanValidator, RuleBasedPlanner, SkillRegistry
from primitives import ManipulationPrimitives
from robot import PandaRobot
from runtime import AgentRuntime, SemanticSkillFactory
from sim import make_environment
from world import WorldModel


def record(output: Path) -> None:
    # One frame per 8 control commands: 20 Hz simulation shown at 10 fps = 4x.
    frames: list[Image.Image] = []
    records = []
    for scenario in ("pick_place", "object_lost", "classical_push"):
        env = make_environment(render=False, seed=67)
        renderer = None
        try:
            env.reset()
            robot, world = PandaRobot(env), WorldModel(env)
            renderer = mujoco.Renderer(env.sim.model._model, height=360, width=640)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.02, 0.03, 0.90]
            camera.distance, camera.azimuth, camera.elevation = 1.25, 135, -45
            option = mujoco.MjvOption()
            option.geomgroup[0] = 0  # Hide collision geometry, as in the GUI demo.
            label = {
                "pick_place": "1 / Pick -> Place | deterministic semantic Agent",
                "object_lost": "2 / Recovery | explicit external cube displacement",
                "classical_push": "3 / Classical Push | right_side region",
            }[scenario]
            status, commands = "Observing fresh simulator state", 0

            def capture() -> None:
                renderer.update_scene(env.sim.data._data, camera=camera, scene_option=option)
                canvas = Image.new("RGB", (640, 444), "#101b29")
                canvas.paste(Image.fromarray(renderer.render()), (0, 44))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 8), label, fill="white")
                draw.text((12, 25), "REAL MUJOCO RUN | 4x simulation speed | state-only", fill="#83cfdd")
                draw.text((12, 412), status[:99], fill="white")
                draw.text((12, 428), f"Control commands: {commands} | no learned checkpoint", fill="#b7c8da")
                frames.append(canvas.resize((560, 388), Image.Resampling.LANCZOS))

            def observe(event) -> None:
                nonlocal commands
                if commands % 8 == 0:
                    capture()
                commands += 1

            trace = []

            def log(message: str) -> None:
                nonlocal status
                trace.append(message)
                if message.startswith("EXECUTE"):
                    skill = message.split(": ", 1)[1].split("(", 1)[0]
                    status = f"Executing {skill}: bounded Cartesian skill"
                elif message.startswith("REPLAN REQUIRED"):
                    status = "Observed object loss -> validate a new Pick / Place plan"
                    capture()
                    frames.extend([frames[-1]] * 9)
                elif message == "AGENT SUCCESS":
                    status = "Goal verified from fresh simulator state"

            disturbance = ObjectLostAfterPick(env) if scenario == "object_lost" else None

            def after_step(event) -> None:
                nonlocal status
                if disturbance is not None and not disturbance.triggered:
                    disturbance(event)
                    if disturbance.triggered:
                        status = "INJECTED: cube moved back to table after Pick (test event)"
                        capture()
                        frames.extend([frames[-1]] * 9)

            registry = SkillRegistry.standard()
            runtime = AgentRuntime(
                planner=RuleBasedPlanner(), registry=registry,
                validator=PlanValidator(registry), observer=world.semantic_state,
                skill_factory=SemanticSkillFactory(world, ManipulationPrimitives(robot)),
                logger=log,
            )
            goal = (
                Goal.push_to_region("Push red cube to right_side", "red_cube", "right_side")
                if scenario == "classical_push" else
                Goal.put_inside("Put red cube inside blue target", "red_cube", "blue_target")
            )
            robot.add_command_observer(observe)
            capture()
            result = runtime.run(goal, after_step=after_step)
            relation = "red_cube_inside_right_side" if scenario == "classical_push" else "red_cube_inside_blue_target"
            satisfied = result.final_world_state.relations[relation]
            status = f"Verified: {relation}={satisfied} | replans={result.replans}"
            capture()
            frames.extend([frames[-1]] * 14)
            records.append({
                "scenario": scenario, "seed": 67, "control_commands": commands,
                "disturbance_triggered": bool(disturbance and disturbance.triggered),
                "result": result.to_dict(),
            })
            if not result.success or not satisfied:
                raise RuntimeError(f"{scenario} did not satisfy its semantic goal")
            if disturbance and (not disturbance.triggered or result.replans != 1):
                raise RuntimeError("recovery clip did not exercise the expected replan")
        finally:
            if renderer is not None:
                renderer.close()
            env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep one palette so the unchanged background remains identical between
    # frames; per-frame adaptive palettes make the GIF much larger.
    palette = frames[0].quantize(colors=256)
    encoded = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    encoded[0].save(output, save_all=True, append_images=encoded[1:], duration=100, loop=0)
    report = {
        "source_algorithm_commit": "1002984923a14ae8c1b466b6845827d3a53a59cd",
        "capture_module": "demos.record_portfolio",
        "capture_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "simulation_speed": "4x",
        "control_hz": 20, "commands_per_frame": 8, "display_fps": 10,
        "frames_before_gif_coalescing": len(frames),
        "output_dimensions": [560, 388],
        "versions": {p: importlib.metadata.version(p) for p in ("mujoco", "robosuite", "numpy", "Pillow")},
        "python": platform.python_version(), "platform": platform.platform(),
        "gif_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "records": records,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "scenarios": [r["scenario"] for r in records], "all_success": True}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/semantic-portfolio.gif"))
    record(parser.parse_args().output)


if __name__ == "__main__":
    main()
