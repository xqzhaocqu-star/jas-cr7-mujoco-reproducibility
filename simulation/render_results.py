#!/usr/bin/env python3
"""Render an offline playback of recorded physical MuJoCo states.

Assigning stored qpos in this module is solely for offline visualization. It
does not run an experiment, integrate dynamics, or alter any recorded result.
The experiments themselves must be produced by torque-driven mj_step.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


def _nearest_indices(t, requested):
    right = np.searchsorted(t, requested, side="left").clip(0, len(t) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(t[left] - requested) < np.abs(t[right] - requested), left, right)


def _line(scene, start, end, color, width=2.):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.asarray(end) - start) < 1e-12:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                       np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
    scene.ngeom += 1


def _caption(bgr, case, timestamp, position_error_mm, orientation_error_deg, trails, mode_caption=""):
    panel_height = 106 if mode_caption else 83
    panel = bgr[:panel_height].copy()
    panel[:] = (14, 18, 24)
    bgr[:panel_height] = cv2.addWeighted(bgr[:panel_height], .15, panel, .85, 0)
    cv2.putText(bgr, f"JAS cooperative transport | {case}", (20, 29),
                cv2.FONT_HERSHEY_SIMPLEX, .68, (245, 245, 245), 1, cv2.LINE_AA)
    if mode_caption:
        cv2.putText(bgr, mode_caption, (20, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    .49, (135, 215, 250), 1, cv2.LINE_AA)
    caption = (f"t = {timestamp:6.3f} s   |   COM error = {position_error_mm:6.2f} mm"
               f"   |   Euler error = {orientation_error_deg:7.4f} deg")
    cv2.putText(bgr, caption, (20, 81 if mode_caption else 58), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (215, 223, 230), 1, cv2.LINE_AA)
    cv2.putText(bgr, "Recorded physical states - offline playback", (20, bgr.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, .43, (205, 215, 225), 1, cv2.LINE_AA)
    if trails:
        cv2.putText(bgr, "COM trails: cyan = actual; gold = reference", (20, 100 if mode_caption else 77),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (205, 215, 225), 1, cv2.LINE_AA)
    return bgr


def render_results(input_path, output_path, model_path=None, fps=30., width=960,
                   height=640, camera="overview", duration=None, trails=False):
    """Create an MP4, first/middle/final montage, and playback metadata JSON."""
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    summary_path = input_path.with_name("summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    settings = summary.get("settings", {})
    mode = settings.get("robust_mode")
    mode_caption = (f"BOUNDARY-LAYER ENGINEERING VARIANT | epsilon = {float(settings['robust_width']):g}"
                    if mode == "boundary_layer" else "PAPER SIGN FEEDBACK" if mode == "paper_sign" else "")
    model_path = Path(model_path or HERE / "models" / "dual_cr7.xml").resolve()
    if fps <= 0 or width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Use a positive fps and positive even video dimensions")
    with np.load(input_path, allow_pickle=False) as saved:
        t, qpos = saved["t"], saved["qpos"]
        qvel = saved["qvel"] if "qvel" in saved else None
        actual, reference = saved["object_pose"], saved["reference"]
        error = saved["error"] if "error" in saved else actual - reference
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("Playback requires strictly increasing recorded timestamps")
    end_time = t[-1] if duration is None else min(t[-1], t[0] + duration)
    if end_time <= t[0]:
        raise ValueError("Playback duration must be positive")
    frames = int(np.floor((end_time - t[0]) * fps + 1e-8)) + 1
    requested = t[0] + np.arange(frames) / fps
    indices = _nearest_indices(t, requested)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer: {output_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if qpos.shape[1] != model.nq:
        writer.release()
        raise ValueError(f"Recorded nq={qpos.shape[1]} differs from model nq={model.nq}")
    data = mujoco.MjData(model)
    snapshots, snapshot_frames = {}, {0, frames // 2, frames - 1}
    start = time.perf_counter()
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for frame, k in enumerate(indices):
                # OFFLINE PLAYBACK ONLY: display the saved physical state. No
                # experiment or simulation step is performed in this module.
                data.qpos[:] = qpos[k]
                if qvel is not None:
                    data.qvel[:] = qvel[k]
                data.time = float(t[k])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                if trails:
                    # Visual trail density is limited to video timestamps; the
                    # scientific result plots retain all controller samples.
                    shown = indices[:frame + 1]
                    for position, rgba in [(actual[shown, :3], [.1, .82, 1., 1.]),
                                           (reference[shown, :3], [1., .65, .10, 1.])]:
                        for a, b in zip(position[:-1], position[1:]):
                            _line(renderer.scene, a, b, rgba)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                bgr = _caption(bgr, input_path.parent.name.replace("_", " "), float(t[k]),
                               float(np.linalg.norm(error[k, :3]) * 1000),
                               float(np.linalg.norm(error[k, 3:]) * 180 / np.pi), trails, mode_caption)
                writer.write(bgr)
                if frame in snapshot_frames:
                    snapshots[frame] = bgr.copy()
                if frame % 90 == 0 or frame == frames - 1:
                    print(f"render {frame + 1}/{frames}, t={t[k]:.3f} s, wall={time.perf_counter()-start:.1f} s", flush=True)
    finally:
        writer.release()
    montage_path = output_path.with_name(output_path.stem + "_montage.png")
    montage = np.concatenate([snapshots[k] for k in sorted(snapshots)], axis=1)
    cv2.imwrite(str(montage_path), montage)
    scene_path = output_path.with_name(output_path.stem + "_scene.png")
    cv2.imwrite(str(scene_path), snapshots[0])
    metadata_path = output_path.with_suffix(".metadata.json")
    info = {"purpose": "Offline visualization of recorded full-dynamics states; no physical rollout is generated here",
            "input_trajectory": str(input_path), "model": str(model_path),
            "output_video": str(output_path), "fps": fps, "width": width, "height": height,
            "frame_count": frames, "first_recorded_time_s": float(t[indices[0]]),
            "last_recorded_time_s": float(t[indices[-1]]),
            "video_container_duration_s": frames / fps, "camera": camera,
            "state_sampling": "nearest recorded state at each video timestamp; no pose interpolation",
            "robust_mode": mode, "mode_caption": mode_caption,
            "boundary_layer_width": settings.get("robust_width") if mode == "boundary_layer" else None,
            "montage": str(montage_path), "initial_scene": str(scene_path),
            "render_wall_seconds": time.perf_counter() - start}
    metadata_path.write_text(json.dumps(info, indent=2) + "\n")
    # Decode the finalized stream, rather than trusting only VideoWriter status.
    check = cv2.VideoCapture(str(output_path))
    ok, decoded = check.read()
    decoded_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if not ok or decoded is None or decoded_frames != frames:
        raise RuntimeError(f"Video verification failed: decoded={ok}, frame count={decoded_frames}, expected={frames}")
    return {"video": output_path, "montage": montage_path, "scene": scene_path, "metadata": metadata_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results" / "nominal" / "trajectory.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path, default=HERE / "models" / "dual_cr7.xml")
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--duration", type=float, help="limit playback to the first N seconds")
    parser.add_argument("--trails", action="store_true", help="optional actual/reference COM line trails")
    args = parser.parse_args()
    paths = render_results(args.input, args.output or args.input.with_name("video.mp4"), args.model,
                           args.fps, args.width, args.height, args.camera, args.duration, args.trails)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
