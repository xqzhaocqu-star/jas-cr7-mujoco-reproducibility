#!/usr/bin/env python3
"""Offline movie adapter with an explicit measured-force extension caption.

The base playback renderer and raw simulation records are left untouched.
Only a completed case with independent validation PASS may be rendered here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import render_results as base

EXTENSION_CAPTION = "EXTRA FORCE SENSING | NEW HISTORY IDENTIFIER"
_ORIGINAL_CAPTION = base._caption


def _extension_caption(bgr, case, timestamp, position_error_mm, orientation_error_deg,
                       trails, mode_caption=""):
    result = _ORIGINAL_CAPTION(bgr, case, timestamp, position_error_mm,
                               orientation_error_deg, trails, mode_caption)
    # The existing robust-mode, clock/error and offline-playback captions stay
    # visible. Add a separate line below their existing top panel.
    strip = result[106:134].copy()
    strip[:] = (14, 18, 24)
    result[106:134] = base.cv2.addWeighted(result[106:134], .15, strip, .85, 0)
    base.cv2.putText(result, EXTENSION_CAPTION, (20, 125), base.cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (135, 245, 190), 1, base.cv2.LINE_AA)
    return result


def render_identification(input_path, output_path, model_path=None, fps=30., width=960,
                          height=640, camera="overview", duration=None, trails=False):
    """Call the original offline playback with a temporary caption adapter."""
    input_path = Path(input_path).resolve()
    summary = json.loads(input_path.with_name("summary.json").read_text())
    validation_path = input_path.with_name("validation.json")
    if not validation_path.exists() or not json.loads(validation_path.read_text()).get("passed", False):
        raise ValueError("The measured-force movie requires completed independent validation PASS")
    settings = summary.get("settings", {})
    if "identification_rate" not in settings:
        raise ValueError("Expected measured-force identification-extension results")
    old_caption = base._caption
    try:
        base._caption = _extension_caption
        paths = base.render_results(input_path, output_path, model_path, fps, width,
                                     height, camera, duration, trails)
    finally:
        base._caption = old_caption
    metadata_path = paths["metadata"]
    metadata = json.loads(metadata_path.read_text())
    metadata.update(algorithm_variant=summary.get("algorithm_variant"),
                    sensing_extension=summary.get("sensing_extension"),
                    learning_extension=summary.get("learning_extension"),
                    identification_rate=float(settings["identification_rate"]),
                    identification_caption=EXTENSION_CAPTION,
                    independent_validation=str(validation_path))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base.HERE/"results_exponential/nominal/trajectory.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path, default=base.HERE/"models/dual_cr7.xml")
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--duration", type=float, help="Limit offline playback to the first N seconds")
    parser.add_argument("--trails", action="store_true")
    args = parser.parse_args()
    paths = render_identification(args.input, args.output or args.input.with_name("video.mp4"),
                                  args.model, args.fps, args.width, args.height,
                                  args.camera, args.duration, args.trails)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
