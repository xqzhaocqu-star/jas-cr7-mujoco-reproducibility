#!/usr/bin/env python3
"""Prepare a paper figure from the real dual-CR7 model and a stored initial state.

Only offline mj_forward and rendering are performed. The plant XML, recorded
trajectory and controller are never edited; no rollout or truth-file read is
performed. The single scene uses a native column width, with vector labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLUE = "#2166ac"
ORANGE = "#d87818"
INK = "#293441"
GREY = "#707b85"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project(point, camera_position, camera_rotation, fovy, width, height):
    local = camera_rotation.T @ (np.asarray(point) - camera_position)
    if local[2] >= 0:
        raise ValueError("Figure annotation lies behind the scene camera")
    f = 1 / np.tan(np.deg2rad(fovy) / 2)
    return np.array([width / 2 + height * f * local[0] / (-2 * local[2]),
                     height / 2 - height * f * local[1] / (-2 * local[2])])


def _render(model_path, state, width=2400, height=2100):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    # Presentation changes to this in-memory rendering model only.
    ground = model.geom("ground").id
    model.geom_rgba[ground] = [.74, .78, .82, 1.]
    data = mujoco.MjData(model)
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    data.time = state["time_s"]
    mujoco.mj_forward(model, data)  # Stored-state display only; no mj_step.
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera="overview", scene_option=option)
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
    # Replace only the renderer's empty sky pixels, preserving physical meshes.
    rgb[segmentation[:, :, 0] < 0] = 255
    foreground = ((segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                  & (segmentation[:, :, 0] != ground))
    yy, xx = np.nonzero(foreground)
    camera = model.camera("overview").id
    camera_position = data.cam_xpos[camera].copy()
    camera_rotation = data.cam_xmat[camera].reshape(3, 3).copy()
    fovy = float(model.cam_fovy[camera])
    positions = {name: data.xpos[model.body(name).id].copy()
                 for name in ("left_base", "right_base", "left_link3", "right_link3", "payload")}
    positions.update({name: data.site_xpos[model.site(name).id].copy()
                      for name in ("left_palm", "right_palm")})
    left = max(0, int(xx.min()) - 150)
    right = min(width, int(xx.max()) + 160)
    top = max(0, int(yy.min()) - 60)
    bottom = min(height, int(yy.max()) + 60)
    crop = rgb[top:bottom, left:right]

    def project(point):
        px, py = _project(point, camera_position, camera_rotation, fovy, width, height)
        return np.array([(px - left) / (right - left), 1 - (py - top) / (bottom - top)])

    return crop, project, positions, dict(
        camera="overview", source_resolution_px=[width, height],
        crop_pixels_xyxy=[left, top, right, bottom],
        embedded_render_resolution_px=[int(crop.shape[1]), int(crop.shape[0])],
        background="Only empty sky composited onto white using MuJoCo segmentation",
        presentation_ground_rgba=[.74, .78, .82, 1.],
        physics_xml_modified=False, simulation_steps=0)


def _callout(ax, label, target, text, color=INK, align="left"):
    ax.annotate(label, xy=target, xytext=text, xycoords="axes fraction", textcoords="axes fraction",
                ha=align, va="center", fontsize=7.4, color=color,
                bbox=dict(boxstyle="round,pad=.18", fc="white", ec="none", alpha=.92),
                arrowprops=dict(arrowstyle="-", lw=.7, color=color,
                                connectionstyle="angle3,angleA=0,angleB=90"))


def _known_geometry(metadata):
    # Geometry remains available for the caption and provenance, not in a panel.
    grasps = metadata["grasp_frames"]
    left = np.asarray(grasps["left"]["base_position_world_m"], dtype=float)
    right = np.asarray(grasps["right"]["base_position_world_m"], dtype=float)
    palms = {h: np.asarray(grasps[h]["palm_position_object_m"], dtype=float)
             for h in ("left", "right")}
    return dict(base_positions_world_m=dict(left=left.tolist(), right=right.tolist()),
                base_separation_m=float(np.linalg.norm(right-left)),
                palm_positions_object_m={h: p.tolist() for h, p in palms.items()},
                grasp_span_m=float(np.linalg.norm(palms["right"]-palms["left"])),
                payload_dimensions_m=(2*np.asarray(metadata["payload_nominal"]["halfsize_m"])).tolist(),
                nominal_com_height_m=float(metadata["payload_nominal"]["initial_position_world_m"][2]))


def prepare_scene(input_path, output_dir, *, dpi=600, render_width=2400, render_height=2100):
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    model_path = HERE / "models" / "dual_cr7.xml"
    metadata_path = HERE / "models" / "scene_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    with np.load(input_path, allow_pickle=False) as raw:
        state = dict(time_s=float(raw["t"][0]), qpos=raw["qpos"][0].copy(),
                     qvel=raw["qvel"][0].copy(), object_pose=raw["object_pose"][0].copy())
    if abs(state["time_s"]) > 1e-12:
        raise ValueError("The model figure must use the recorded t=0 initial state")
    manifest_path = input_path.with_name("source_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    model_digest = _sha256(model_path)
    if manifest["simulation/models/dual_cr7.xml"] != model_digest:
        raise ValueError("Current model differs from the recorded experiment manifest")
    image, project, positions, render_info = _render(model_path, state, render_width, render_height)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    figure_size = [3.5, 2.4]
    fig = plt.figure(figsize=figure_size, facecolor="white")
    scene_ax = fig.add_axes([.01, .01, .98, .98])
    scene_ax.imshow(image, extent=(0, 1, 0, 1),
                    aspect=image.shape[0] / image.shape[1], interpolation="lanczos")
    scene_ax.set_xlim(0, 1)
    scene_ax.set_ylim(0, 1)
    scene_ax.axis("off")
    _callout(scene_ax, "Left CR7", project(positions["left_link3"]), (.015, .34), BLUE)
    _callout(scene_ax, "Right CR7", project(positions["right_link3"]), (.98, .72), ORANGE, "right")
    _callout(scene_ax, "Rigid payload", project(positions["payload"] + [0., 0., .06]), (.31, .965))
    for hand, color, label, text in (("left", BLUE, "$c_L$", (.40, .57)),
                                     ("right", ORANGE, "$c_R$", (.78, .91))):
        p = project(positions[hand + "_palm"])
        scene_ax.plot(*p, "o", color=color, mec="white", mew=.6, ms=4.2, zorder=5)
        _callout(scene_ax, label, p, text, color)
    origin = project([0., 0., 0.])
    for direction, color, label in (([.19, 0, 0], "#b44640", "$x_W$"),
                                     ([0, .19, 0], "#25835f", "$y_W$"),
                                     ([0, 0, .19], "#3e69a7", "$z_W$")):
        end = project(direction)
        scene_ax.annotate("", xy=end, xytext=origin,
                          arrowprops=dict(arrowstyle="-|>", lw=.9, color=color, mutation_scale=7))
        scene_ax.text(*(end + [.012, .012]), label, fontsize=7.8, color=color)
    scene_ax.text(*(origin + [-.025, -.04]), "$O_W$", fontsize=7.4, color=GREY)
    geometry = _known_geometry(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "dual_cr7_simulation_model"
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, facecolor="white")
    plt.close(fig)
    info = dict(purpose="Single-column paper model figure: real MuJoCo scene with vector labels",
                input_trajectory=str(input_path), input_trajectory_sha256=_sha256(input_path),
                model=str(model_path), model_sha256=model_digest,
                metadata=str(metadata_path), metadata_sha256=_sha256(metadata_path),
                source_manifest=str(manifest_path), script=str(Path(__file__).resolve()),
                script_sha256=_sha256(__file__), unknown_parameter_truth_read=False,
                mesh_asset_manifest=str(HERE / "models" / "asset_sha256.json"),
                mesh_asset_manifest_sha256=_sha256(HERE / "models" / "asset_sha256.json"),
                state_time_s=state["time_s"], recorded_initial_object_pose_m_rad=state["object_pose"].tolist(),
                layout="Single scene; independent nominal top-view geometry panel removed",
                layout_change="Replaced two-panel 5.6 x 2.85 in layout by native single-column 3.5 x 2.4 in scene",
                geometry_panel=False,
                omitted_in_figure="Dimensions, COM heights and finite-compliance weld note are provided in the manuscript caption/text",
                geometry=geometry, render=render_info, figure_size_inches=figure_size,
                figure_size_mm=(np.asarray(figure_size) * 25.4).tolist(), raster_export_dpi=dpi,
                suggested_tex="\\begin{figure}[t]\\centering\\includegraphics[width=\\columnwidth]{manuscript_figures/dual_cr7_simulation_model.pdf}\\caption{...}\\end{figure}",
                recommended_layout="Single-column figure at 3.45 to 3.5 inches (87.63 to 88.90 mm) width, using columnwidth",
                minimum_main_label_fontsize_pt=7.4,
                minimum_main_label_fontsize_at_3_45in_pt=7.4 * 3.45 / figure_size[0],
                pdf_content="High-resolution actual render with vector labels and projected world axes; not wholly vector",
                output_png=str(stem.with_suffix(".png")), output_pdf=str(stem.with_suffix(".pdf")),
                caption_suggestion=("Dual-xMate CR7 cooperative MuJoCo model at the recorded nominal initial state. "
                    "Markers c_L and c_R locate the payload grasps. The 0.28 x 0.40 x 0.12 m payload "
                    "is held by finite-compliance welds; base and grasp separations are 1.30 and 0.40 m. "
                    "The recorded and nominal COM heights are 0.638 and 0.650 m, respectively."))
    info["output_sha256"] = {suffix: _sha256(stem.with_suffix(suffix)) for suffix in (".pdf", ".png")}
    stem.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results_comparison/main/nominal/proposed/trajectory.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "manuscript_figures")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    result = prepare_scene(args.input, args.output_dir, dpi=args.dpi)
    print(json.dumps({k: result[k] for k in ("output_png", "output_pdf", "figure_size_mm")}, indent=2))


if __name__ == "__main__":
    main()
