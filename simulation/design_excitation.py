#!/usr/bin/env python3
"""Offline excitation design using nominal CR7 inertia bases and known priors.

No plant XML, plant truth, recorded parameter values or physical rollout is
loaded. Only a small JSON of reference feasibility and nominal Gram spectra
is written. Accelerations are known reference derivatives used offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from jas.barrier import transform
from jas.dynamics import LinkDynamics, object_dynamics
from jas.geometry import LocalKinematics
from jas.reference import desired as existing_reference, envelope

HERE = Path(__file__).resolve().parent
PROFILES = {
    "moderate": {"amplitude": [.055, .040, .035, .20, .16, .22],
                 "angular_frequency": [1.7, 2.1, 2.7, 3., 3.7, 4.1]},
    "rich": {"amplitude": [.065, .045, .045, .23, .18, .25],
             "angular_frequency": [2.2, 2.7, 3.1, 3.7, 4.3, 5.1]},
    "rotation_rich": {"amplitude": [.065, .035, .035, .26, .22, .30],
                      "angular_frequency": [1.9, 2.5, 3., 4., 4.7, 5.6]},
    "compact_fast": {"amplitude": [.03575, .026, .02275, .13, .104, .143],
                     "angular_frequency": [3.06, 3.78, 4.86, 5.4, 6.66, 7.38]},
}
HARMONIC_MULTIPLIERS = np.array([1., 1.83, 2.71])
HARMONIC_WEIGHTS = np.array([.65, .25, .10])
PHASES = np.array([[0., 1.1, 2.4], [.7, 2.2, .3], [1.2, .2, 1.8],
                   [.4, 1.7, 2.8], [1., 2.6, .6], [1.8, .5, 2.]])


def multisine_reference(t, center, duration=20., profile="moderate"):
    """C2 startup, six-axis three-tone motion, and C2 phase slowdown to hold."""
    spec = PROFILES[profile]
    startup = 2.
    if t < startup:
        z = max(t, 0.) / startup
        gate = 10*z**3-15*z**4+6*z**5
        dgate = (30*z**2-60*z**3+30*z**4)/startup
        ddgate = (60*z-180*z**2+120*z**3)/startup**2
    else:
        gate, dgate, ddgate = 1., 0., 0.
    stop_start = duration - 2.
    if t <= stop_start:
        phase, speed, acceleration = t, 1., 0.
    elif t < duration:
        z = (t-stop_start)/2.
        phase = stop_start+2*(z-2.5*z**4+3*z**5-z**6)
        speed = 1-10*z**3+15*z**4-6*z**5
        acceleration = (-30*z**2+60*z**3-30*z**4)/2.
    else:
        phase, speed, acceleration = duration-1., 0., 0.
    amplitude = np.asarray(spec["amplitude"])[:, None] * HARMONIC_WEIGHTS
    omega = np.asarray(spec["angular_frequency"])[:, None] * HARMONIC_MULTIPLIERS
    argument = omega*phase+PHASES
    wave = np.sum(amplitude*np.sin(argument), axis=1)
    first = np.sum(amplitude*omega*np.cos(argument), axis=1)*speed
    second = np.sum(amplitude*(-omega**2*np.sin(argument)*speed**2
                               +omega*np.cos(argument)*acceleration), axis=1)
    return (np.asarray(center)+gate*wave, dgate*wave+gate*first,
            ddgate*wave+2*dgate*first+gate*second)


def _kinematics(metadata, hand):
    frame = metadata["grasp_frames"][hand]
    base_rotation = Rotation.from_euler("z", frame["base_yaw_rad"]).as_matrix()
    palm_rotation = np.asarray(frame["palm_rotation_object"])
    offset = -palm_rotation.T@np.asarray(frame["palm_position_object_m"])
    return LocalKinematics(HERE/"models/arm_nominal.xml", frame["base_position_world_m"],
                           base_rotation, offset, palm_rotation.T)


class FastNominalBases:
    """Known per-body inertia-basis torque, independent of unknown densities."""
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(HERE/"models/arm_nominal.xml"))
        self.data = mujoco.MjData(self.model)
        self.bodies = [self.model.body("link"+str(j)).id for j in range(1, 7)]
        self.jp = np.zeros((3, 6)); self.jr = np.zeros((3, 6))
        self.dp = np.zeros((3, 6)); self.dr = np.zeros((3, 6))

    def torque(self, q, qd, qdd):
        m, d = self.model, self.data
        d.qpos[:] = q; d.qvel[:] = qd
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_comVel(m, d)
        result = np.zeros((6, 6))
        for j, body in enumerate(self.bodies):
            mujoco.mj_jacBodyCom(m, d, self.jp, self.jr, body)
            mujoco.mj_jacDot(m, d, self.dp, self.dr, d.xipos[body], body)
            rotation = d.ximat[body].reshape(3, 3)
            inertia = (rotation*m.body_inertia[body])@rotation.T
            omega = self.jr@qd
            force = m.body_mass[body]*(self.jp@qdd+self.dp@qd-m.opt.gravity)
            moment = inertia@(self.jr@qdd+self.dr@qd)+np.cross(omega, inertia@omega)
            result[j] = self.jp.T@force+self.jr.T@moment
        return result


def _verify_fast_bases():
    fast = FastNominalBases(); full = LinkDynamics(HERE/"models/arm_nominal.xml")
    maximum = 0.
    for n in range(5):
        q = np.array([.5, 1.3, 2., -.65, -2., -1.9])+.05*np.sin(n+np.arange(6))
        qd = .4*np.sin(.7*n+np.arange(6)); qdd = .8*np.cos(.9*n+np.arange(6))
        mass, coriolis, gravity = full.evaluate(q, qd)
        reference = np.einsum("aij,j->ai", mass, qdd)+np.einsum("aij,j->ai", coriolis, qd)+gravity
        maximum = max(maximum, float(np.max(np.abs(fast.torque(q, qd, qdd)-reference))))
    if maximum > 1e-7:
        raise RuntimeError(f"Independent nominal dynamics basis mismatch: {maximum}")
    return maximum


def _solve_pose(kinematics, target, initial):
    q = initial.copy()
    for _ in range(20):
        pose, jacobian = kinematics.pose_jacobian(q)
        error = target-pose
        if np.max(np.abs(error)) < 5e-11:
            return q, jacobian
        increment = np.linalg.solve(jacobian, error)
        increment *= min(1., .25/max(np.max(np.abs(increment)), 1e-15))
        q += increment
    raise ValueError("Continuous nominal inverse-kinematics branch did not close")


def _gram_information(gram, learning_scale=1., gamma=.2):
    gram = .5*(gram+gram.T)
    diagonal = np.diag(gram)
    scale = 1./np.sqrt(np.maximum(diagonal, 1e-20))
    normalized = (scale[:, None]*gram)*scale[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    normalized_eigenvalues = np.linalg.eigvalsh(normalized)
    kappa = learning_scale
    upsilon = kappa/np.maximum(diagonal, 1e-20)
    rates = gamma*kappa*normalized_eigenvalues
    return {"eigenvalues": eigenvalues.tolist(), "diagonal": diagonal.tolist(),
            "condition_number": float(eigenvalues[-1]/max(eigenvalues[0], 1e-30)),
            "normalized_eigenvalues": normalized_eigenvalues.tolist(),
            "normalized_condition_number": float(normalized_eigenvalues[-1]/max(normalized_eigenvalues[0], 1e-30)),
            "weakest_raw_parameter_direction": eigenvectors[:, 0].tolist(),
            "gram": gram.tolist(), "normalized_gram": normalized.tolist(),
            "upsilon_diagonal_for_kappa": upsilon.tolist(),
            "kappa": kappa, "gamma": gamma, "nominal_homogeneous_learning_rates_s_inv": rates.tolist()}


def _window_information(times, integral, dt):
    windows = {}
    for window in (2., 4., 6., 8.):
        if window > times[-1]:
            continue
        count = int(round(window/dt))
        grams = integral[count:]-integral[:-count]
        first = _gram_information(grams[0])
        normalized_minima = []
        for n in range(0, len(grams), max(1, int(round(.1/dt)))):
            gram = grams[n]; diag = 1./np.sqrt(np.maximum(np.diag(gram), 1e-20))
            normalized_minima.append((float(np.linalg.eigvalsh((diag[:, None]*gram)*diag[None, :])[0]), n))
        _, best_index = max(normalized_minima)
        windows[f"{window:g}"] = {"first_window_end_s": float(times[count]), "first": first,
                                 "best_normalized_window_end_s": float(times[count+best_index]),
                                 "best": _gram_information(grams[best_index])}
    return windows


def evaluate_profile(profile, duration=20., dt=.01, filter_rate=8.):
    metadata = json.loads((HERE/"models/scene_metadata.json").read_text())
    center = np.r_[metadata["payload_nominal"]["initial_position_world_m"], np.zeros(3)]
    reference = ((lambda t: existing_reference(t, center)) if profile == "existing"
                 else (lambda t: multisine_reference(t, center, duration, profile)))
    times = np.arange(int(round(duration/dt))+1)*dt
    chi = np.array([20., 20., 20., 8., 8., 8.]); ell = np.array([50., 50., 50., 20., 20., 20.])
    limits = np.asarray(metadata["torque_limits_Nm"])
    velocity_limits = np.asarray(metadata["joint_velocity_limits_rad_s"])
    joint_limits = np.asarray(metadata["joint_position_limits_rad"])
    payload = metadata["payload_nominal"]
    output = {"profile": profile, "duration_s": duration, "design_grid_dt_s": dt,
              "filter_rate_s_inv": filter_rate, "hands": {},
              "input_access": ["arm_nominal.xml", "scene_metadata.json", "known C2 reference", "known parameter prior box"],
              "not_a_rollout": True}
    positions = np.array([reference(t)[0] for t in times])
    output["maximum_absolute_task_offset_m_rad"] = np.max(np.abs(positions-center), axis=0).tolist()
    for hand_index, hand in enumerate(("left", "right")):
        kin = _kinematics(metadata, hand); dyn = FastNominalBases()
        q = np.asarray(metadata["initial_ik"][hand]["q"])
        qmin = np.full(6, np.inf); qmax = -qmin.copy()
        max_qd = np.zeros(6); max_qdd = np.zeros(6)
        max_torque = np.zeros(6); max_prior_torque = np.zeros(6)
        minimum_sigma = np.inf; maximum_condition = 0.; maximum_pose_error = 0.
        filtered = np.zeros((8, 6)); old_regressor = None; old_product = np.zeros((8, 8))
        integral = np.zeros((len(times), 8, 8))
        robot_filtered = np.zeros((7, 6)); old_robot = None; old_robot_product = np.zeros((7, 7))
        robot_integral = np.zeros((len(times), 7, 7))
        payload_filtered = np.zeros((1, 6)); old_payload = None; old_payload_product = np.zeros((1, 1))
        payload_integral = np.zeros((len(times), 1, 1))
        decay = np.exp(-filter_rate*dt)
        ramp = 1-(1-decay)/(filter_rate*dt)
        desired_force = np.array([0., (-1 if hand_index == 0 else 1)*5., 0., 0., 0., 0.])
        for n, t in enumerate(times):
            x, xd, xdd = reference(t)
            q, jacobian = _solve_pose(kin, x, q)
            qd = np.linalg.solve(jacobian, xd)
            pose, jacobian, jdot = kin.evaluate(q, qd)
            qdd = np.linalg.solve(jacobian, xdd-jdot@qd)
            robot = dyn.torque(q, qd, qdd)
            mo, co, go = object_dynamics(x, xd, payload["mass_kg"], payload["diagonal_inertia_kg_m2"])
            bases = np.vstack((robot, .05*qd, .5*jacobian.T@(mo@xdd+co@xd+go)))
            robot_regressor = bases[:7]
            payload_regressor = (mo@xdd+co@xd+go)[None, :]
            bs = transform(np.zeros(6), np.zeros(6), envelope(t, chi, ell), chi, ell)
            matrix_a = np.linalg.inv(jacobian)/bs.a
            regressor = bases@matrix_a
            if old_regressor is not None:
                filtered = decay*filtered+(1-decay)*old_regressor+ramp*(regressor-old_regressor)
                product = filtered@filtered.T
                integral[n] = integral[n-1]+.5*dt*(old_product+product)
                old_product = product
                robot_filtered = decay*robot_filtered+(1-decay)*old_robot+ramp*(robot_regressor-old_robot)
                robot_product = robot_filtered@robot_filtered.T
                robot_integral[n] = robot_integral[n-1]+.5*dt*(old_robot_product+robot_product)
                old_robot_product = robot_product
                payload_filtered = decay*payload_filtered+(1-decay)*old_payload+ramp*(payload_regressor-old_payload)
                payload_product = payload_filtered@payload_filtered.T
                payload_integral[n] = payload_integral[n-1]+.5*dt*(old_payload_product+payload_product)
                old_payload_product = payload_product
            old_regressor = regressor
            old_robot = robot_regressor
            old_payload = payload_regressor
            torque = np.sum(bases, axis=0)-jacobian.T@desired_force
            # Conservative corner bound for all eight independent [.85,1.15] parameters.
            prior_torque = np.abs(torque)+.15*np.sum(np.abs(bases), axis=0)
            qmin = np.minimum(qmin, q); qmax = np.maximum(qmax, q)
            max_qd = np.maximum(max_qd, np.abs(qd)); max_qdd = np.maximum(max_qdd, np.abs(qdd))
            max_torque = np.maximum(max_torque, np.abs(torque))
            max_prior_torque = np.maximum(max_prior_torque, prior_torque)
            singular = np.linalg.svd(jacobian, compute_uv=False)
            minimum_sigma = min(minimum_sigma, float(singular[-1]))
            maximum_condition = max(maximum_condition, float(singular[0]/singular[-1]))
            maximum_pose_error = max(maximum_pose_error, float(np.max(np.abs(pose-x))))
        windows = _window_information(times, robot_integral, dt)
        margin = np.minimum(qmin-joint_limits[:, 0], joint_limits[:, 1]-qmax)
        output["hands"][hand] = {"q_min_rad": qmin.tolist(), "q_max_rad": qmax.tolist(),
                                  "minimum_joint_margin_rad": float(margin.min()),
                                  "minimum_jacobian_sigma": minimum_sigma, "maximum_jacobian_condition": maximum_condition,
                                  "maximum_nominal_IK_pose_error": maximum_pose_error,
                                  "maximum_joint_speed_rad_s": max_qd.tolist(),
                                  "maximum_joint_acceleration_rad_s2": max_qdd.tolist(),
                                  "maximum_joint_speed_limit_ratio": float(np.max(max_qd/velocity_limits)),
                                  "maximum_nominal_feedforward_torque_Nm": max_torque.tolist(),
                                  "maximum_prior_box_feedforward_torque_bound_Nm": max_prior_torque.tolist(),
                                  "maximum_prior_box_feedforward_torque_limit_ratio": float(np.max(max_prior_torque/limits)),
                                  "robot_window_grams": windows,
                                  "coupled_window_grams": _window_information(times, integral, dt),
                                  "payload_window_grams": _window_information(times, payload_integral, dt),
                                  "nominal_feasibility_screen_pass": bool(
                                      margin.min() > 0 and minimum_sigma > .04
                                      and np.max(max_qd/velocity_limits) < 1
                                      and np.max(max_prior_torque/limits) < 1)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=("existing", *PROFILES), default=["existing", "moderate"])
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--output", type=Path, default=HERE/"identification_design/design_summary.json")
    args = parser.parse_args()
    if args.duration < 10. or args.dt <= 0 or args.dt > .05:
        parser.error("Use duration>=10 s and 0<dt<=0.05 s")
    began = time.perf_counter()
    result = {"purpose": __doc__, "parameter_order": ["link1", "link2", "link3", "link4", "link5", "link6", "viscous", "payload"],
              "robot_parameter_order": ["link1", "link2", "link3", "link4", "link5", "link6", "viscous"],
              "reference_definition": {"profiles": PROFILES,
                                       "harmonic_multipliers": HARMONIC_MULTIPLIERS.tolist(),
                                       "harmonic_weights": HARMONIC_WEIGHTS.tolist(),
                                       "phases_rad": PHASES.tolist(), "startup_s": 2., "slowdown_s": 2.},
              "recommendation": {"profile": "moderate", "duration_s": 20.,
                                 "filter_rate_s_inv": 8., "window_s": 6.,
                                 "nominal_raw_robot_gram_threshold": .003,
                                 "prior_theta_interval": [.85, 1.15],
                                 "diagonal_gain_rule": "Upsilon = kappa * diag(G)^{-1}; conditioning limits remain",
                                 "caution": "Nominal feasibility only; full Gram preconditioning is an explicitly separate engineering extension."},
              "fast_basis_vs_existing_finite_difference_maximum_error": _verify_fast_bases(),
              "profiles": {}}
    for profile in args.profiles:
        item = evaluate_profile(profile, args.duration, args.dt)
        result["profiles"][profile] = item
        print(profile, {hand: {"min_sigma": record["minimum_jacobian_sigma"],
                               "speed_ratio": record["maximum_joint_speed_limit_ratio"],
                               "prior_torque_ratio": record["maximum_prior_box_feedforward_torque_limit_ratio"],
                               "G8_normalized_min": record["robot_window_grams"]["8"]["first"]["normalized_eigenvalues"][0]}
                        for hand, record in item["hands"].items()}, flush=True)
    result["wall_seconds"] = time.perf_counter()-began
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Nominal-only design diagnostics: {args.output}", flush=True)


if __name__ == "__main__":
    main()
