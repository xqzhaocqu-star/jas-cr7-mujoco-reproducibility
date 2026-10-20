"""Rebuild the two physical CR7 arms from audited TRO robot assets only.

No TRO task, controller, or reference trajectory is imported.  The nominal model
is a single 6-DoF arm whose six inertial bodies scale linearly with six density
parameters; a seventh independent scalar scales all six viscous dampings. The
plant has two such arms and a free rigid payload, joined only
through two 6D weld constraints.  ``plant_truth.json`` is plant-only information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
TORQUE_LIMITS = np.array([194., 194., 102., 66., 66., 29.])
SPEED_LIMITS = np.deg2rad([180., 180., 234., 240., 240., 300.])
POSITION_LIMIT = 3.0543
PALM_OFFSET = np.array([0., 0., .10])
PAYLOAD_POSITION = np.array([0., 0., .65])
PAYLOAD_HALFSIZE = np.array([.14, .20, .06])
PAYLOAD_MASS = 2.
BASES = {"left": (np.array([0., -.65, 0.]), np.pi / 2),
         "right": (np.array([0., .65, 0.]), -np.pi / 2)}
GRASPS = {"left": (np.array([0., -.20, 0.]), Rotation.from_euler("x", -np.pi / 2).as_matrix()),
          "right": (np.array([0., .20, 0.]), Rotation.from_euler("x", np.pi / 2).as_matrix())}
TRUE_SCALES = {"left": np.array([.94, 1.06, 1.04, .97, 1.05, .96]),
               "right": np.array([1.06, .94, .96, 1.03, .95, 1.04]),
               "left_viscous": .98, "right_viscous": 1.04,
               "payload": 1.08}


def _s(x):
    return " ".join(f"{v:.15g}" for v in np.asarray(x).ravel())


def _quat(R):
    q = Rotation.from_matrix(R).as_quat()
    return np.r_[q[3], q[:3]]


def _write_xml(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _header(name):
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets", inertiafromgeom="false", autolimits="true")
    ET.SubElement(root, "option", timestep="0.00025", integrator="implicitfast", gravity="0 0 -9.81",
                  cone="elliptic", solver="Newton", iterations="100", tolerance="1e-12")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.05", armature="0", frictionloss="0", limited="true")
    ET.SubElement(default, "geom", friction="0.6 0.005 0.0001", solref="0.005 1", solimp="0.95 0.99 0.001")
    asset = ET.SubElement(root, "asset")
    for part in ["base"] + [f"link{i}" for i in range(1, 7)]:
        ET.SubElement(asset, "mesh", name=f"cr7_{part}", file=f"xMateCR7_{part}.stl", scale="0.001 0.001 0.001", inertia="convex")
    return root


def _add_arm(world, actuator, props, prefix="", base=(0., 0., 0.), yaw=0., scales=None, viscous_scale=1.):
    scales = np.ones(6) if scales is None else np.asarray(scales)
    current = ET.SubElement(world, "body", name=f"{prefix}base", pos=_s(base), quat=_s(_quat(Rotation.from_euler("z", yaw).as_matrix())))
    base_props = props["xMateCR7_base"]
    ET.SubElement(current, "inertial", pos=_s(base_props["com_body_frame_m"]),
                  quat=_s(base_props["principal_frame_quaternion_wxyz"]), mass=str(base_props["mass_kg"]),
                  diaginertia=_s(base_props["principal_inertia_kg_m2"]))
    ET.SubElement(current, "geom", name=f"{prefix}mesh_base", type="mesh", mesh="cr7_base", mass="0", contype="2", conaffinity="1", rgba="0.80 0.83 0.87 1")
    positions = [(0, 0, 0), (0, 0, .296), (0, 0, .49), (0, 0, .36), (0, -.151, 0), (0, 0, .1265)]
    axes = [(0, 0, 1), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, -1, 0), (0, 0, 1)]
    for k in range(6):
        i = k + 1
        p = props[f"xMateCR7_link{i}"]
        current = ET.SubElement(current, "body", name=f"{prefix}link{i}", pos=_s(positions[k]))
        ET.SubElement(current, "joint", name=f"{prefix}joint{i}", type="hinge", axis=_s(axes[k]), range=_s([-POSITION_LIMIT, POSITION_LIMIT]), damping=f"{.05 * viscous_scale:.15g}")
        ET.SubElement(current, "inertial", pos=_s(p["com_body_frame_m"]), quat=_s(p["principal_frame_quaternion_wxyz"]),
                      mass=f"{p['mass_kg'] * scales[k]:.15g}", diaginertia=_s(np.array(p["principal_inertia_kg_m2"]) * scales[k]))
        ET.SubElement(current, "geom", name=f"{prefix}mesh_link{i}", type="mesh", mesh=f"cr7_link{i}", mass="0", contype="2", conaffinity="1",
                      rgba="0.34 0.39 0.46 1" if i in (4, 6) else "0.80 0.83 0.87 1")
        ET.SubElement(actuator, "motor", name=f"{prefix}torque{i}", joint=f"{prefix}joint{i}", gear="1", ctrllimited="true",
                      ctrlrange=_s([-TORQUE_LIMITS[k], TORQUE_LIMITS[k]]), forcelimited="true", forcerange=_s([-TORQUE_LIMITS[k], TORQUE_LIMITS[k]]))
    # All bracket geoms are visual only and add exactly zero mass/inertia.
    ET.SubElement(current, "geom", name=f"{prefix}grasp_bracket", type="cylinder", pos="0 0 0.05", size="0.024 0.05", mass="0", contype="0", conaffinity="0", rgba="0.95 0.66 0.13 1")
    ET.SubElement(current, "geom", name=f"{prefix}grasp_pad", type="box", pos=_s(PALM_OFFSET - [0, 0, .004]), size="0.06 0.035 0.004", mass="0", contype="0", conaffinity="0", rgba="0.20 0.25 0.29 1")
    ET.SubElement(current, "site", name=f"{prefix}palm", pos=_s(PALM_OFFSET), size="0.008", rgba="0.9 0.2 0.1 1")


def _ik_candidates(model, position, rotation, seed=10):
    data = mujoco.MjData(model)
    site_id = model.site("palm").id
    rng = np.random.default_rng(seed)

    def residual(q):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        er = Rotation.from_matrix(rotation @ data.site_xmat[site_id].reshape(3, 3).T).as_rotvec()
        return np.r_[data.site_xpos[site_id] - position, .5 * er]

    result = []
    seeds = [np.array([0, .7, -1.5, 0, -1.2, 0]), np.array([0, -.7, 1.5, 0, 1.2, 0])]
    seeds += [rng.uniform(-2.7, 2.7, 6) for _ in range(36)]
    for q0 in seeds:
        sol = least_squares(residual, q0, bounds=(-POSITION_LIMIT + .04, POSITION_LIMIT - .04), max_nfev=180,
                            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if np.linalg.norm(residual(sol.x)) > 1e-7:
            continue
        jp, jr = np.zeros((3, 6)), np.zeros((3, 6))
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        sigma = np.linalg.svd(np.vstack((jp, jr)), compute_uv=False)[-1]
        if sigma < .04:
            continue
        margin = POSITION_LIMIT - np.max(np.abs(sol.x))
        # Resolve branch choices by Jacobian conditioning, then source-limit margin.
        score = sigma + .015 * margin
        result.append((score, sol.x.copy(), sigma, margin))
    if not result:
        raise RuntimeError("No admissible nonsingular grasp IK solution")
    return sorted(result, key=lambda x: x[0], reverse=True)


def _initial_configs(model):
    info = {}
    for n, (base, yaw) in BASES.items():
        offset, Rp = GRASPS[n]
        Rb = Rotation.from_euler("z", yaw).as_matrix()
        candidate = _ik_candidates(model, Rb.T @ (PAYLOAD_POSITION + offset - base), Rb.T @ Rp, seed=10)
        _, q, sigma, margin = candidate[0]
        info[n] = {"q": q.tolist(), "sigma_min": float(sigma), "joint_limit_margin_rad": float(margin)}
    return info


def build_scene():
    """Write both MJCFs plus separate prior and truth JSON; return their paths."""
    MODELS.mkdir(parents=True, exist_ok=True)
    audit = json.loads((MODELS / "source" / "TRO_model_audit.json").read_text())
    props = {p["name"]: p for p in audit["mass_properties"]}
    nominal = _header("CR7_single_arm_nominal_no_tool")
    world = ET.SubElement(nominal, "worldbody")
    actuator = ET.SubElement(nominal, "actuator")
    _add_arm(world, actuator, props)
    nominal_path = MODELS / "arm_nominal.xml"
    _write_xml(nominal, nominal_path)
    nominal_model = mujoco.MjModel.from_xml_path(str(nominal_path))
    initial = _initial_configs(nominal_model)

    plant = _header("JAS_dual_xMateCR7_rigid_payload")
    visual = ET.SubElement(plant, "visual")
    ET.SubElement(visual, "global", offwidth="1440", offheight="1000")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")
    world = ET.SubElement(plant, "worldbody")
    ET.SubElement(world, "light", pos="1 -1 2.6", dir="-0.3 0.2 -1", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", pos="0 0 -0.02", size="2 2 .05", contype="1", conaffinity="6", rgba="0.19 0.23 0.28 1")
    ET.SubElement(world, "camera", name="overview", pos="1.8 -1.7 1.65", xyaxes="0.686 0.728 0 -0.314 0.296 0.902", fovy="43")
    ET.SubElement(world, "camera", name="front", pos="2.0 0 1.15", xyaxes="0 1 0 -0.243 0 0.970", fovy="45")
    actuator = ET.SubElement(plant, "actuator")
    equality = ET.SubElement(plant, "equality")
    grasp_frames = {}
    for name, (base, yaw) in BASES.items():
        _add_arm(world, actuator, props, prefix=f"{name}_", base=base, yaw=yaw, scales=TRUE_SCALES[name], viscous_scale=TRUE_SCALES[f"{name}_viscous"])
        off, Rp = GRASPS[name]
        flange_offset = off - Rp @ PALM_OFFSET
        relative_payload_pos = -Rp.T @ flange_offset
        relative_payload_quat = _quat(Rp.T)
        ET.SubElement(equality, "weld", name=f"{name}_rigid_grasp", body1=f"{name}_link6", body2="payload",
                      relpose=_s(np.r_[relative_payload_pos, relative_payload_quat]), solref="0.002 1", solimp="0.999 0.9999 0.0001", torquescale="0.2")
        grasp_frames[name] = {"base_position_world_m": base.tolist(), "base_yaw_rad": float(yaw),
                              "palm_position_object_m": off.tolist(), "palm_rotation_object": Rp.tolist(),
                              "flange_position_object_m": flange_offset.tolist(),
                              "payload_position_flange_m": relative_payload_pos.tolist(),
                              "payload_quaternion_flange_wxyz": relative_payload_quat.tolist()}
    payload = ET.SubElement(world, "body", name="payload", pos=_s(PAYLOAD_POSITION))
    ET.SubElement(payload, "freejoint", name="payload_free")
    inertia = PAYLOAD_MASS / 3 * np.array([PAYLOAD_HALFSIZE[1] ** 2 + PAYLOAD_HALFSIZE[2] ** 2,
                                         PAYLOAD_HALFSIZE[0] ** 2 + PAYLOAD_HALFSIZE[2] ** 2,
                                         PAYLOAD_HALFSIZE[0] ** 2 + PAYLOAD_HALFSIZE[1] ** 2])
    ET.SubElement(payload, "inertial", pos="0 0 0", mass=str(PAYLOAD_MASS * TRUE_SCALES["payload"]), diaginertia=_s(inertia * TRUE_SCALES["payload"]))
    ET.SubElement(payload, "geom", name="payload_box", type="box", size=_s(PAYLOAD_HALFSIZE), mass="0", contype="4", conaffinity="1", rgba="0.23 0.58 0.74 1")
    ET.SubElement(payload, "site", name="object_center", size=".008", rgba="1 0.3 0.2 1")
    q_initial = np.r_[initial["left"]["q"], initial["right"]["q"], PAYLOAD_POSITION, [1, 0, 0, 0]]
    keyframe = ET.SubElement(plant, "keyframe")
    ET.SubElement(keyframe, "key", name="initial_grasp", qpos=_s(q_initial))
    plant_path = MODELS / "dual_cr7.xml"
    _write_xml(plant, plant_path)
    m = mujoco.MjModel.from_xml_path(str(plant_path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    errors = {}
    for name, (off, Rp) in GRASPS.items():
        site_id = m.site(f"{name}_palm").id
        errors[name] = {"position_m": float(np.linalg.norm(d.site_xpos[site_id] - PAYLOAD_POSITION - off)),
                        "orientation_rad": float(Rotation.from_matrix(d.site_xmat[site_id].reshape(3, 3) @ Rp.T).magnitude())}
    if max(v["position_m"] for v in errors.values()) > 1e-6 or np.linalg.norm(d.efc_pos) > 1e-6:
        raise RuntimeError(f"Inconsistent grasp initialization: {errors}, eq {d.efc_pos}")
    metadata = {"schema_version": 1, "mujoco_version": mujoco.__version__, "plant_xml": plant_path.name,
                "nominal_arm_xml": nominal_path.name, "qpos_order": ["left_joint1..6", "right_joint1..6", "payload_xyz_wxyz"],
                "dof_order": ["left_joint1..6", "right_joint1..6", "payload_translation_world", "payload_rotation_local"],
                "q_initial": q_initial.tolist(), "initial_ik": initial, "initial_grasp_error": errors,
                "initial_equality_residual_norm": float(np.linalg.norm(d.efc_pos)), "initial_contact_count": int(d.ncon),
                "arms": ["left", "right"], "grasp_frames": grasp_frames,
                "palm_offset_flange_m": PALM_OFFSET.tolist(),
                "parameterization": {"arm_parameters": "six independent density multipliers per arm; each scales body mass and COM inertia only",
                                     "arm_prior_lower": [.85] * 6, "arm_prior_upper": [1.15] * 6,
                                     "viscous_parameter": "one independent common scalar multiplying all six nominal joint viscous dampings per arm",
                                     "viscous_prior_lower": .85, "viscous_prior_upper": 1.15,
                                     "payload_parameter": "one density multiplier on centered uniform box mass and inertia", "payload_prior_lower": .85, "payload_prior_upper": 1.15,
                                     "local_parameter_order": ["link1_density", "link2_density", "link3_density", "link4_density", "link5_density", "link6_density", "common_viscous_scale", "payload_density"],
                                     "local_prior_lower": [.85] * 8, "local_prior_upper": [1.15] * 8,
                                     "nominal_viscous_damping_Nm_s_rad": [.05] * 6, "armature_kg_m2": [0.] * 6},
                "payload_nominal": {"mass_kg": PAYLOAD_MASS, "halfsize_m": PAYLOAD_HALFSIZE.tolist(), "com_object_m": [0, 0, 0],
                                    "diagonal_inertia_kg_m2": inertia.tolist(), "initial_position_world_m": PAYLOAD_POSITION.tolist(), "initial_quaternion_wxyz": [1, 0, 0, 0]},
                "torque_limits_Nm": TORQUE_LIMITS.tolist(), "joint_velocity_limits_rad_s": SPEED_LIMITS.tolist(),
                "joint_position_limits_rad": [[-POSITION_LIMIT, POSITION_LIMIT]] * 6,
                "physics_timestep": {"default_s": .00025, "runtime_choices_s": [.00025, .000125], "note": "runtime may set model.opt.timestep before integration; controller period is independently configured"},
                "contacts": "source TRO self-collision masks retained; arm-arm and arm-payload collision suppressed; arms and payload contact ground; rigid-grasp load transfer uses weld equality constraints",
                "inertia_provenance": "TRO convex-hull uniform-density mesh approximations; explicit inertials copied from compiled audit, not unreliable original xacro inertia",
                "layout_adjustments": "none; requested +/-0.65 m bases and z=0.65 m payload retained"}
    metadata_path = MODELS / "scene_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    truth_path = MODELS / "plant_truth.json"
    truth_path.write_text(json.dumps({"controller_access": "for offline evaluation and plant construction only; controller must use prior bounds, not true multipliers",
                                      "left_density_multipliers": TRUE_SCALES["left"].tolist(), "right_density_multipliers": TRUE_SCALES["right"].tolist(),
                                      "left_viscous_multiplier": TRUE_SCALES["left_viscous"], "right_viscous_multiplier": TRUE_SCALES["right_viscous"],
                                      "payload_density_multiplier": TRUE_SCALES["payload"]}, indent=2) + "\n")
    hashes = {str(p.relative_to(MODELS)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((MODELS / "assets").glob("*.stl"))}
    hashes.update({str(p.relative_to(MODELS)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((MODELS / "source").glob("*")) if p.is_file()})
    (MODELS / "asset_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return {"plant": plant_path, "nominal": nominal_path, "metadata": metadata_path, "plant_truth": truth_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="render the actual initial physical model")
    args = parser.parse_args()
    paths = build_scene()
    if args.render:
        m = mujoco.MjModel.from_xml_path(str(paths["plant"]))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        from PIL import Image
        with mujoco.Renderer(m, height=1000, width=1440) as renderer:
            renderer.update_scene(d, camera="overview")
            Image.fromarray(renderer.render()).save(MODELS / "initial_scene.png")
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
