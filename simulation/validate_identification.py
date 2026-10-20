#!/usr/bin/env python3
"""Independently audit the explicitly measured-force identification extension.

Only validation.json and validation_summary.json are written. Raw trajectories,
learning records, sources, and models are read only. The exact exponential
check concerns the frozen data solution; its separately measured bias from
physical truth is always reported, without relabelling that bias as zero.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from validate_results import (Audit, CASES, HERE, PROJECT, compare_reference,
                              independent_bounds, independent_reference,
                              maximum_error, plain, reconstruct_kinematics,
                              replay_intervals)


def verify_sources(audit, manifest):
    required = {'bare_jrnl_new_sample4.tex', 'simulation/run_identification.py',
                'simulation/run_experiments.py', 'simulation/design_excitation.py',
                'simulation/models/dual_cr7.xml', 'simulation/models/arm_nominal.xml',
                'simulation/models/scene_metadata.json', 'simulation/models/plant_truth.json'}
    for folder in ('jas', 'exponential_identification'):
        required.update(str(path.relative_to(PROJECT)) for path in (HERE/folder).glob('*.py'))
    missing, mismatches = sorted(required-set(manifest)), []
    for relative, expected in manifest.items():
        path = PROJECT/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(relative)
    audit.check('source_manifest_sha256', not missing and not mismatches,
                checked_files=len(manifest), missing_required_sources=missing,
                mismatches=mismatches)
    assets = json.loads((HERE/'models/asset_sha256.json').read_text())
    bad_assets = [relative for relative, expected in assets.items()
                  if not (HERE/'models'/relative).is_file()
                  or hashlib.sha256((HERE/'models'/relative).read_bytes()).hexdigest() != expected]
    audit.check('imported_asset_sha256', not bad_assets,
                checked_files=len(assets), mismatches=bad_assets)
    # This is supplemental source inspection, not a replacement for the
    # independent dynamic and held-command checks below.
    tree = ast.parse((HERE/'exponential_identification/regression.py').read_text())
    forbidden = {'qacc', 'qM', 'qfrc_bias', 'body_mass', 'body_inertia', 'dof_damping'}
    reads = sorted({node.attr for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute) and node.attr in forbidden})
    audit.check('online_regression_has_no_acceleration_or_truth_dynamics_reads', not reads,
                forbidden_attributes_read=reads,
                scope='AST inspection of measured regression implementation; nominal bases are caller inputs.')


def configured_model(summary):
    model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
    model.opt.timestep = float(summary['physics_dt_s'])
    name = summary.get('integrator', summary.get('integrator_s', 'implicitfast'))
    if isinstance(name, str):
        name = name.rsplit('_', 1)[-1].upper()
        model.opt.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_'+name)
    else:
        model.opt.integrator = int(name)
    return model


def independent_task_reference(times, center, summary):
    """Position-level reference check independent of the runner's evaluator."""
    profile = summary.get('reference_profile', summary['settings']['excitation_profile'])
    if profile == 'baseline':
        return independent_reference(times, center)
    # Profile constants are design inputs, not fitted data or plant truth.
    from design_excitation import PROFILES
    spec = PROFILES[profile]
    frequencies = np.asarray(spec['angular_frequency'])[:, None]*np.array([1., 1.83, 2.71])
    amplitudes = np.asarray(spec['amplitude'])[:, None]*np.array([.65, .25, .10])
    shifts = np.array([[0., 1.1, 2.4], [.7, 2.2, .3], [1.2, .2, 1.8],
                       [.4, 1.7, 2.8], [1., 2.6, .6], [1.8, .5, 2.]])
    phase = np.asarray(times).copy()
    duration = min(float(summary['duration_s']),
                   float(summary['settings'].get('excitation_duration', summary['duration_s'])))
    transition = (times > duration-2.) & (times < duration)
    z = (times[transition]-(duration-2.))/2.
    phase[transition] = duration-2.+2*(z-2.5*z**4+3*z**5-z**6)
    phase[times >= duration] = duration-1.
    z = np.clip(times/2., 0., 1.)
    gate = 10*z**3-15*z**4+6*z**5
    waves = np.sum(amplitudes[None, :, :]*np.sin(
        phase[:, None, None]*frequencies[None, :, :]+shifts[None, :, :]), axis=2)
    return np.asarray(center)+gate[:, None]*waves


def validate_force_filter(audit, arrays, summary):
    """Reintegrate the measured-force low-pass with both command-side limits."""
    times = arrays['t']
    k = float(summary['settings']['filter_rate'])
    left, right = arrays['measured_force_left'], arrays['measured_force_right']
    expected = np.zeros_like(arrays['filtered_torque'])
    for n, dt in enumerate(np.diff(times), start=1):
        decay = np.exp(-k*dt)
        new_weight = 1+np.expm1(-k*dt)/(k*dt)
        old_weight = -np.expm1(-k*dt)-new_weight
        expected[n] = decay*expected[n-1]+old_weight*right[n-1]+new_weight*left[n]
    error = maximum_error(expected, arrays['filtered_torque'])
    audit.check('force_filter_reconstructed_with_held_command_endpoint_limits', error <= 2e-9,
                maximum_error=error, tolerance=2e-9,
                interval_samples='previous right limit to current left limit; right jump starts next interval')
    if 'measured_momentum' in arrays and 'measured_remainder' in arrays:
        momentum, remainder = arrays['measured_momentum'], arrays['measured_remainder']
        source = remainder-k*momentum
        state = np.zeros_like(momentum[0])
        maximum = maximum_error(np.zeros_like(state), arrays['filtered_regressor'][0])
        for n, dt in enumerate(np.diff(times), start=1):
            decay = np.exp(-k*dt)
            new_weight = 1+np.expm1(-k*dt)/(k*dt)
            old_weight = -np.expm1(-k*dt)-new_weight
            state = decay*state+old_weight*source[n-1]+new_weight*source[n]
            expected_y = k*(momentum[n]-np.exp(-k*times[n])*momentum[0])+state
            maximum = max(maximum, maximum_error(expected_y, arrays['filtered_regressor'][n]))
        audit.check('acceleration_free_regressor_filter_reconstructed_from_raw_momentum_signals', maximum <= 2e-9,
                    maximum_error=maximum, tolerance=2e-9,
                    initial_condition='Analytic exp(-k*t) initial-momentum subtraction retained.')


def validate_learning(audit, arrays, saved, summary, truth):
    """Raw quadrature, IE acceptance, historical solution, and exponential law."""
    times = arrays['t']
    yf, force = arrays['filtered_regressor'], arrays['filtered_torque']
    integrand_g = yf@np.swapaxes(yf, -1, -2)
    integrand_p = np.einsum('napc,nac->nap', yf, force)
    cg, cp = np.zeros_like(integrand_g), np.zeros_like(integrand_p)
    dt = np.diff(times)
    cg[1:] = np.cumsum(.5*dt[:, None, None, None]*(integrand_g[1:]+integrand_g[:-1]), axis=0)
    cp[1:] = np.cumsum(.5*dt[:, None, None]*(integrand_p[1:]+integrand_p[:-1]), axis=0)
    cfg = summary['settings']
    window, threshold = float(cfg['identification_window']), float(cfg['identification_threshold'])
    rate = float(cfg['identification_rate'])
    lower, upper = float(cfg['prior_lower']), float(cfg['prior_upper'])
    cutoff = np.maximum(0., times-window)
    left = np.searchsorted(times, cutoff, side='right')-1
    grams, ps = cg-cg[left], cp-cp[left]
    grams = .5*(grams+np.swapaxes(grams, -1, -2))
    raw_lambdas = np.linalg.eigvalsh(grams)[..., 0]
    diagonal = np.diagonal(grams, axis1=2, axis2=3)
    valid_diagonal = np.all(diagonal > 1e-14, axis=2)
    scales = 1/np.sqrt(np.maximum(diagonal, 1e-300))
    normalized = (scales[:, :, :, None]*grams)*scales[:, :, None, :]
    normalized_lambdas = np.zeros_like(raw_lambdas)
    normalized_lambdas[valid_diagonal] = np.linalg.eigvalsh(normalized[valid_diagonal])[..., 0]
    detected, final_g, final_p, targets, freeze_thetas = [], [], [], [], []
    expected_lambdas, expected_normalized = raw_lambdas.copy(), normalized_lambdas.copy()
    expected_prediction = np.empty_like(arrays['prediction_error'])
    expected_theta = np.full_like(arrays['theta'], .9)
    rejected_counts, rates_measured, contraction_errors, floor_bounds = [], [], [], []
    for arm in range(2):
        candidates = np.flatnonzero((times >= window) & (raw_lambdas[:, arm] > 0)
                                    & (normalized_lambdas[:, arm] >= threshold)
                                    & valid_diagonal[:, arm])
        accepted, target, rejected = None, None, 0
        for index in candidates:
            candidate_target = np.linalg.solve(grams[index, arm], ps[index, arm])
            if np.all((candidate_target >= lower) & (candidate_target <= upper)):
                accepted, target = int(index), candidate_target
                break
            rejected += 1
        rejected_counts.append(rejected)
        detected.append(float(times[accepted]) if accepted is not None else None)
        endpoint = accepted if accepted is not None else len(times)-1
        final_g.append(grams[endpoint, arm]); final_p.append(ps[endpoint, arm])
        targets.append(np.full(8, np.nan) if target is None else target)
        active_g, active_p = grams[:, arm].copy(), ps[:, arm].copy()
        if accepted is not None:
            active_g[accepted:] = active_g[accepted]
            active_p[accepted:] = active_p[accepted]
            expected_lambdas[accepted:, arm] = raw_lambdas[accepted, arm]
            expected_normalized[accepted:, arm] = normalized_lambdas[accepted, arm]
            theta0 = arrays['theta'][accepted, arm]
            freeze_thetas.append(theta0)
            decay = np.exp(-rate*(times[accepted:]-times[accepted]))
            expected_theta[accepted:, arm] = target+decay[:, None]*(theta0-target)
            deviation = np.linalg.norm(arrays['theta'][accepted:, arm]-target, axis=1)
            eligible = (deviation > 1e-8) & (times[accepted:]-times[accepted] >= .05)
            if np.count_nonzero(eligible) >= 4:
                slope = np.polyfit(times[accepted:][eligible]-times[accepted],
                                   np.log(deviation[eligible]), 1)[0]
                rates_measured.append(float(-slope))
            else:
                rates_measured.append(None)
            contraction_errors.append(maximum_error(arrays['theta'][accepted:, arm],
                                                     expected_theta[accepted:, arm]))
            truth_error = np.linalg.norm(arrays['theta'][accepted:, arm]-truth[arm], axis=1)
            floor = float(np.linalg.norm(target-truth[arm]))
            bound = decay*np.linalg.norm(theta0-truth[arm])+(1-decay)*floor
            floor_bounds.append(float(np.max(truth_error-bound)))
        else:
            freeze_thetas.append(np.full(8, np.nan)); rates_measured.append(None)
            contraction_errors.append(None); floor_bounds.append(None)
        expected_prediction[:, arm] = active_p-np.einsum('npq,nq->np', active_g,
                                                        arrays['theta_control'][:, arm])
    final_g, final_p, targets = np.array(final_g), np.array(final_p), np.array(targets)
    recovered_ok = all(a is not None and b is not None and abs(a-b) <= 1e-9
                       for a, b in zip(detected, summary['ie_detection_times_s']))
    audit.check('ie_first_accepted_window_reconstructed_from_raw_data', recovered_ok
                and rejected_counts == summary['rejected_outside_prior_counts'],
                recomputed_times_s=detected, reported_times_s=summary['ie_detection_times_s'],
                window_s=window, normalized_eigenvalue_threshold=threshold,
                rejected_outside_prior_counts=rejected_counts,
                reported_rejections=summary['rejected_outside_prior_counts'],
                acceptance='Raw SPD Gram; diagonally normalized lambda >= threshold; solved data target within prior.')
    g_error, p_error = maximum_error(final_g, saved['gram']), maximum_error(final_p, saved['p'])
    g_tol = 5e-9+2e-13*float(np.max(np.abs(final_g)))
    p_tol = 5e-9+2e-13*float(np.max(np.abs(final_p)))
    audit.check('stored_gram_and_p_are_unregularized_raw_integrals', g_error <= g_tol and p_error <= p_tol,
                maximum_gram_error=g_error, gram_tolerance=g_tol,
                maximum_p_error=p_error, p_tolerance=p_tol,
                raw_minimum_eigenvalues=np.linalg.eigvalsh(final_g)[:, 0])
    audit.check('raw_and_normalized_ie_eigenvalues_preserved_after_freeze',
                maximum_error(expected_lambdas, arrays['lambda_min']) <= 5e-8
                and maximum_error(expected_normalized, arrays['normalized_lambda_min']) <= 2e-9,
                maximum_raw_lambda_error=maximum_error(expected_lambdas, arrays['lambda_min']),
                maximum_normalized_lambda_error=maximum_error(expected_normalized, arrays['normalized_lambda_min']))
    prediction_error = maximum_error(expected_prediction, arrays['prediction_error'])
    audit.check('historical_prediction_uses_control_estimate_before_update', prediction_error <= 2e-7,
                maximum_error=prediction_error, tolerance=2e-7)
    target_error = maximum_error(targets, saved['data_target'])
    audit.check('data_target_equals_unregularized_gram_solve_p',
                np.all(np.isfinite(targets)) and target_error <= 3e-8
                and maximum_error(targets, summary['data_target']) <= 3e-8
                and np.all((targets >= lower) & (targets <= upper)),
                maximum_target_error=target_error, recomputed_target=targets,
                solver='Independent numpy.linalg.solve of raw eight-parameter Gram.')
    freeze_error = maximum_error(np.array(freeze_thetas), saved['theta_at_freeze'])
    theta_error = maximum_error(expected_theta, arrays['theta'])
    observed_rates_ok = all(r is not None and abs(r-rate) <= 1e-5 for r in rates_measured)
    audit.check('exponential_contraction_to_data_target_at_prescribed_rate',
                recovered_ok and theta_error <= 3e-8 and freeze_error <= 1e-13 and observed_rates_ok
                and summary['numerical_projection_counts'] == [0, 0],
                configured_rate_s_inv=rate, independently_fitted_rates_s_inv=rates_measured,
                maximum_full_trajectory_parameter_discrepancy=theta_error,
                per_arm_after_freeze_maximum_discrepancy=contraction_errors,
                maximum_theta_at_freeze_discrepancy=freeze_error,
                reported_parameter_projection_counts=summary['numerical_projection_counts'],
                interpretation='Exact sampled exponential law to the measured-data target, not necessarily exact truth.')
    residual = final_p-np.einsum('apq,aq->ap', final_g, truth)
    bias_from_residual = np.stack([np.linalg.solve(final_g[a], residual[a]) for a in range(2)])
    floor_error = maximum_error(targets-truth, bias_from_residual)
    target_floor = np.linalg.norm(targets-truth, axis=1)
    final_error = np.linalg.norm(arrays['theta'][-1]-truth, axis=1)
    audit.check('true_error_separately_decomposed_into_exponential_term_and_bias_floor',
                recovered_ok and floor_error <= 3e-8
                and all(value is not None and value <= 3e-8 for value in floor_bounds)
                and maximum_error(target_floor, summary['data_target_error_norms']) <= 3e-8
                and maximum_error(final_error, summary['final_parameter_error_norms']) <= 1e-13,
                data_target_true_parameter_error_norms=target_floor,
                final_true_parameter_error_norms=final_error,
                maximum_bias_identity_discrepancy=floor_error,
                per_arm_maximum_excess_over_exponential_plus_floor_bound=floor_bounds,
                interpretation='Nonzero numerical bias is retained. Exact truth convergence requires unbiased history.')
    audit.check('stored_residual_is_p_minus_raw_gram_times_truth',
                maximum_error(residual, saved['stored_residual']) <= p_tol+8*g_tol
                and maximum_error(saved['true_parameters_offline_only'], truth) <= 1e-13,
                maximum_residual_error=maximum_error(residual, saved['stored_residual']),
                residual_norms=np.linalg.norm(residual, axis=1))
    # Freeze data are never used in this predictive check: evaluate later raw
    # filtered measurements with both physical truth and the frozen solution.
    heldout = []
    for arm in range(2):
        mask = np.zeros(len(times), dtype=bool) if detected[arm] is None else times > detected[arm]+.5
        if np.any(mask):
            values = {}
            for label, theta in (('truth_offline_only', truth[arm]), ('frozen_data_target', targets[arm])):
                error = force[mask, arm]-np.einsum('npc,p->nc', yf[mask, arm], theta)
                values[label] = dict(robot_vector_rms_Nm=float(np.sqrt(np.mean(np.sum(error[:, :6]**2, axis=1)))),
                                     payload_force_vector_rms_N=float(np.sqrt(np.mean(np.sum(error[:, 6:9]**2, axis=1)))),
                                     payload_moment_vector_rms_Nm=float(np.sqrt(np.mean(np.sum(error[:, 9:]**2, axis=1)))),
                                     per_channel_rms=np.sqrt(np.mean(error**2, axis=0)))
            heldout.append(dict(arm=arm, samples=int(np.count_nonzero(mask)),
                                first_time_s=float(times[mask][0]), last_time_s=float(times[mask][-1]), **values))
        else:
            heldout.append(dict(arm=arm, samples=0))
    audit.check('independent_post_freeze_holdout_residuals_reported',
                all(item['samples'] > 0 for item in heldout), arms=heldout,
                scope='Descriptive out-of-window residuals; mixed force/torque channels are reported separately.')


class NominalNewtonEuler:
    """Independent nominal per-link force basis, without Christoffel matrices."""
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(HERE/'models/arm_nominal.xml'))
        self.data = mujoco.MjData(self.model)

    def forces(self, q, qd, qdd):
        m, d = self.model, self.data
        d.qpos[:] = q; d.qvel[:] = qd
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_comVel(m, d)
        jp, jr, dp, dr = [np.zeros((3, 6)) for _ in range(4)]
        result = np.empty((6, 6))
        for link in range(6):
            body = m.body(f'link{link+1}').id
            mujoco.mj_jacBodyCom(m, d, jp, jr, body)
            mujoco.mj_jacDot(m, d, dp, dr, d.xipos[body], body)
            rotation = d.ximat[body].reshape(3, 3)
            inertia = (rotation*m.body_inertia[body])@rotation.T
            omega = jr@qd
            force = m.body_mass[body]*(jp@qdd+dp@qd-m.opt.gravity)
            moment = inertia@(jr@qdd+dr@qd)+np.cross(omega, inertia@omega)
            result[link] = jp.T@force+jr.T@moment
        return result


def validate_sensors_and_momentum(audit, model, arrays, metadata, truth, samples=32):
    """Offline qacc is allowed only here as an independent dynamics reference."""
    nominal = NominalNewtonEuler()
    payload = metadata['payload_nominal']
    m0 = float(payload['mass_kg']); i0 = np.diag(payload['diagonal_inertia_kg_m2'])
    indices = np.unique(np.linspace(0, len(arrays['t'])-1, min(samples, len(arrays['t']))).astype(int))
    sensor_error, robot_error, payload_error, power_error = 0., 0., 0., 0.
    constraints_error, left_error, right_error = 0., 0., 0.
    raw_momentum_error, raw_remainder_error = 0., 0.
    from jas.dynamics import LinkDynamics
    nominal_mass = LinkDynamics(HERE/'models/arm_nominal.xml')
    payload_body = model.body('payload').id
    for tick in indices:
        data = mujoco.MjData(model)
        data.qpos[:] = arrays['qpos'][tick]; data.qvel[:] = arrays['qvel'][tick]
        data.time = arrays['t'][tick]
        for side in ('left', 'right'):
            command = (np.zeros(12) if tick == 0 else arrays['torque_applied'][tick-1]) if side == 'left' else arrays['torque_applied'][tick]
            data.ctrl[:] = command
            mujoco.mj_forward(model, data)
            rotation = data.xmat[payload_body].reshape(3, 3)
            wrench = np.r_[data.qfrc_constraint[12:15], rotation@data.qfrc_constraint[15:18]]
            measured = np.concatenate((np.reshape(data.qfrc_actuator[:12]+data.qfrc_constraint[:12], (2, 6)),
                                       np.broadcast_to(wrench, (2, 6))), axis=1)
            discrepancy = maximum_error(measured, arrays[f'measured_force_{side}'][tick])
            sensor_error = max(sensor_error, discrepancy)
            if side == 'left':
                left_error = max(left_error, discrepancy)
            else:
                right_error = max(right_error, discrepancy)
            # Independent physical generalized balance checks reaction sign.
            full_mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, data, full_mass)
            balance = full_mass@data.qacc+data.qfrc_bias-data.qfrc_passive-data.qfrc_actuator
            constraints_error = max(constraints_error, maximum_error(balance, data.qfrc_constraint))
            for arm in range(2):
                s = slice(6*arm, 6*arm+6)
                bases = nominal.forces(data.qpos[s], data.qvel[s], data.qacc[s])
                prediction = truth[arm, :6]@bases+truth[arm, 6]*.05*data.qvel[s]
                robot_error = max(robot_error, maximum_error(prediction, measured[arm, :6]))
            if side == 'right' and 'measured_momentum' in arrays:
                # Reconstruct each known momentum basis independently of the
                # observe() helper. The remainder uses the kinetic-energy
                # coordinate gradient, not the controller's C-transpose.
                expected_momentum, expected_remainder = np.zeros((2, 8, 12)), np.zeros((2, 8, 12))
                for arm in range(2):
                    s = slice(6*arm, 6*arm+6)
                    mass_bases, gravity = nominal_mass.masses(data.qpos[s], gravity=True)
                    expected_momentum[arm, :6, :6] = np.einsum('pij,j->pi', mass_bases, data.qvel[s])
                    gradient = np.empty((6, 6))
                    for coordinate in range(6):
                        shift = np.zeros(6); shift[coordinate] = 2e-6
                        plus, _ = nominal_mass.masses(data.qpos[s]+shift)
                        minus, _ = nominal_mass.masses(data.qpos[s]-shift)
                        gradient[:, coordinate] = .5*np.einsum('i,pij,j->p', data.qvel[s],
                                                                 (plus-minus)/(4e-6), data.qvel[s])
                    expected_remainder[arm, :6, :6] = gravity-gradient
                    expected_remainder[arm, 6, :6] = .05*data.qvel[s]
                world_inertia = rotation@i0@rotation.T
                expected_momentum[:, 7, 6:] = np.r_[m0*data.qvel[12:15],
                                                     world_inertia@(rotation@data.qvel[15:18])]
                expected_remainder[:, 7, 6:] = np.r_[-m0*model.opt.gravity, np.zeros(3)]
                raw_momentum_error = max(raw_momentum_error, maximum_error(expected_momentum, arrays['measured_momentum'][tick]))
                raw_remainder_error = max(raw_remainder_error, maximum_error(expected_remainder, arrays['measured_remainder'][tick]))
            inertia_world = rotation@i0@rotation.T
            omega = rotation@data.qvel[15:18]
            alpha = rotation@data.qacc[15:18]
            prediction = truth[0, 7]*np.r_[m0*(data.qacc[12:15]-model.opt.gravity),
                                           inertia_world@alpha+np.cross(omega, inertia_world@omega)]
            payload_error = max(payload_error, maximum_error(prediction, wrench))
            power_error = max(power_error, abs(data.qfrc_constraint[12:]@data.qvel[12:]
                                                -wrench@np.r_[data.qvel[12:15], omega]))
    audit.check('measured_robot_and_payload_forces_reconstructed_at_both_command_limits',
                sensor_error <= 2e-7, checked_states=len(indices),
                maximum_left_held_old_input_error=left_error,
                maximum_right_new_input_error=right_error,
                threshold=2e-7, payload_COM_offset_nominal=payload['com_object_m'])
    audit.check('independent_full_dynamics_reaction_balance', constraints_error <= 2e-7,
                checked_command_side_states=2*len(indices), maximum_generalized_balance_error=constraints_error)
    audit.check('independent_nominal_link_momentum_equation_matches_measured_force', robot_error <= 2e-6,
                checked_robot_equations=4*len(indices), maximum_torque_residual_Nm=robot_error,
                reference='Offline Newton-Euler per-link basis and validation-only plant qacc; no controller Christoffel implementation.')
    audit.check('actual_payload_world_COM_momentum_equation_matches_measured_wrench', payload_error <= 2e-6,
                checked_payload_states=2*len(indices), maximum_wrench_component_residual=payload_error,
                maximum_sensor_coordinate_power_discrepancy_W=power_error,
                reference='Independent world force and angular momentum balance with validation-only qacc and truth.')
    if 'measured_momentum' in arrays:
        audit.check('raw_nominal_momentum_and_remainder_signals_reconstructed_independently',
                    raw_momentum_error <= 2e-9 and raw_remainder_error <= 2e-8,
                    checked_states=len(indices), maximum_momentum_basis_error=raw_momentum_error,
                    maximum_remainder_basis_error=raw_remainder_error,
                    reference='Nominal COM Jacobian mass basis; independent kinetic-energy gradient for C-transpose velocity.')


def validate_case(directory, samples=101, reference=None, expected_duration=None):
    audit = Audit(directory.name)
    audit.report['scope'] = ('All saved ticks, independently reconstructed FK/barriers/history; '
                              '101 short full-dynamics replays by default; 32 states with both command-side '
                              'force and independent dynamic identities. Exponential claim is to the data '
                              'target, with numerical true-parameter floor explicitly reported.')
    required_files = ('trajectory.npz', 'learning_state.npz', 'summary.json', 'source_manifest.json')
    missing = [name for name in required_files if not (directory/name).is_file()]
    audit.check('complete_artifact_files', not missing, missing=missing)
    if missing:
        return audit.finish(directory)
    summary = json.loads((directory/'summary.json').read_text())
    verify_sources(audit, json.loads((directory/'source_manifest.json').read_text()))
    metadata = json.loads((HERE/'models/scene_metadata.json').read_text())
    plant = json.loads((HERE/'models/plant_truth.json').read_text())
    truth = np.array([plant[n+'_density_multipliers']+[plant[n+'_viscous_multiplier'],
                     plant['payload_density_multiplier']] for n in ('left', 'right')])
    with np.load(directory/'trajectory.npz', allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    with np.load(directory/'learning_state.npz', allow_pickle=False) as source:
        learned = {key: source[key] for key in source.files}
    needed = {'t', 'qpos', 'qvel', 'object_pose', 'reference', 'error', 'local_error',
              'torque_command', 'torque_applied', 'surface', 'xi', 'barrier_margin',
              'upper', 'lower', 'sigma_min', 'theta', 'theta_control', 'filtered_regressor',
              'filtered_torque', 'lambda_min', 'normalized_lambda_min', 'prediction_error',
              'measured_force_left', 'measured_force_right', 'weld_residual', 'contact_count'}
    absent = sorted(needed-set(arrays))
    audit.check('required_trajectory_arrays', not absent, missing=absent)
    if absent:
        return audit.finish(directory)
    times = arrays['t']; n = len(times); intervals = np.diff(times)
    dt, physics_dt = float(summary['control_dt_s']), float(summary['physics_dt_s'])
    duration = float(summary['duration_s'] if expected_duration is None else expected_duration)
    grid_ok = (n > 1 and abs(times[0]) <= 1e-12 and abs(times[-1]-duration) <= 1e-9
               and n == round(duration/dt)+1 and np.all(intervals > 0)
               and np.max(np.abs(intervals-dt)) <= 1e-10)
    audit.check('monotonic_complete_time_grid', grid_ok, samples=n, duration_s=duration,
                first_time_s=float(times[0]), last_time_s=float(times[-1]), control_dt_s=dt)
    shapes = {'qpos': (n, 19), 'qvel': (n, 18), 'object_pose': (n, 6),
              'local_error': (n, 2, 6), 'theta': (n, 2, 8), 'theta_control': (n, 2, 8),
              'filtered_regressor': (n, 2, 8, 12), 'filtered_torque': (n, 2, 12),
              'measured_force_left': (n, 2, 12), 'measured_force_right': (n, 2, 12)}
    wrong = {key: dict(actual=arrays[key].shape, expected=shape) for key, shape in shapes.items()
             if arrays[key].shape != shape}
    nonfinite = [key for key in needed-{'upper', 'lower'} if not np.all(np.isfinite(arrays[key]))]
    audit.check('finite_states_and_expected_array_shapes', not wrong and not nonfinite,
                wrong_shapes=wrong, nonfinite_arrays=nonfinite)
    if wrong or nonfinite or n < 2:
        return audit.finish(directory)
    raw_shapes = {key: (n, 2, 8, 12) for key in ('measured_momentum', 'measured_remainder')}
    missing_raw = [key for key in raw_shapes if key not in arrays]
    wrong_raw = {key: arrays[key].shape for key, shape in raw_shapes.items()
                 if key in arrays and (arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])))}
    audit.check('complete_raw_acceleration_free_observation_signals', not missing_raw and not wrong_raw,
                missing=missing_raw, wrong_or_nonfinite_shapes=wrong_raw)
    if wrong_raw:
        return audit.finish(directory)
    quat_error = float(np.max(np.abs(np.linalg.norm(arrays['qpos'][:, 15:19], axis=1)-1)))
    audit.check('payload_unit_quaternion', quat_error <= 1e-10, maximum_norm_error=quat_error)
    center = np.r_[metadata['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    ref = independent_task_reference(times, center, summary)
    audit.check('reference_matches_independently_evaluated_task', maximum_error(ref, arrays['reference']) <= 2e-12,
                maximum_error=maximum_error(ref, arrays['reference']), profile=summary['reference_profile'])
    model = configured_model(summary)
    object_pose, local_pose, sigmas = reconstruct_kinematics(model, arrays, metadata)
    object_error, local_error = object_pose-ref, local_pose-ref[:, None, :]
    audit.check('object_pose_and_errors_reconstructed_from_qpos',
                maximum_error(object_pose, arrays['object_pose']) <= 2e-11
                and maximum_error(object_error, arrays['error']) <= 2e-11,
                maximum_pose_error=maximum_error(object_pose, arrays['object_pose']),
                maximum_error_discrepancy=maximum_error(object_error, arrays['error']))
    audit.check('local_encoder_errors_reconstructed_from_full_scene_fk',
                maximum_error(local_error, arrays['local_error']) <= 2e-10,
                maximum_error=maximum_error(local_error, arrays['local_error']), checked_ticks=n)
    position_gap = np.linalg.norm(local_pose[:, :, :3]-object_pose[:, None, :3], axis=2)
    rotations = Rotation.from_euler('xyz', object_pose[:, 3:])
    angular_gap = np.stack([(Rotation.from_euler('xyz', local_pose[:, arm, 3:])*rotations.inv()).magnitude()
                            for arm in range(2)], axis=1)
    audit.check('numerical_grasp_and_weld_pose_consistency',
                np.max(position_gap) <= 5e-4 and np.max(angular_gap) <= .002
                and np.max(np.abs(arrays['weld_residual'])) <= 1e-3,
                maximum_position_gap_m=float(position_gap.max()),
                maximum_orientation_gap_rad=float(angular_gap.max()),
                maximum_weld_residual=float(np.abs(arrays['weld_residual']).max()))
    chi, ell = np.asarray(summary['settings']['chi']), np.asarray(summary['settings']['ell'])
    mu_a, mu_b = independent_bounds(times, chi, ell, summary['scenario']['nonmonotonic'])
    eta = ell*np.tanh(chi*local_error)
    gap = np.minimum((mu_a[:, None, :]-eta)/ell, (mu_b[:, None, :]+eta)/ell)
    eta_object = ell*np.tanh(chi*object_error)
    object_gap = np.minimum((mu_a-eta_object)/ell, (mu_b+eta_object)/ell)
    audit.check('local_and_actual_object_barriers_admissible_every_tick', np.min(gap) > 0 and np.min(object_gap) > 0,
                minimum_local_gap=float(gap.min()), minimum_actual_object_gap=float(object_gap.min()))
    audit.check('logged_barrier_gap_and_transformed_error',
                maximum_error(gap, arrays['barrier_margin']) <= 2e-10
                and maximum_error(eta/((mu_a[:, None, :]-eta)*(mu_b[:, None, :]+eta)), arrays['xi']) <= 2e-9,
                maximum_gap_discrepancy=maximum_error(gap, arrays['barrier_margin']))
    with np.errstate(divide='ignore'):
        upper, lower = np.arctanh(mu_a/ell)/chi, -np.arctanh(mu_b/ell)/chi
    bounds_ok = (np.all(np.isposinf(arrays['upper'][0])) and np.all(np.isneginf(arrays['lower'][0]))
                 and maximum_error(upper[1:], arrays['upper'][1:]) <= 2e-10
                 and maximum_error(lower[1:], arrays['lower'][1:]) <= 2e-10)
    errors_inside = (np.all(local_error < arrays['upper'][:, None, :])
                     and np.all(local_error > arrays['lower'][:, None, :])
                     and np.all(object_error < arrays['upper']) and np.all(object_error > arrays['lower']))
    audit.check('physical_error_envelopes_at_every_tick', bounds_ok and errors_inside,
                independently_reconstructed_bounds_match=bounds_ok, all_errors_inside=errors_inside)
    sigma_error = maximum_error(sigmas, arrays['sigma_min'])
    minimum_cos = float(np.min(np.abs(np.cos(local_pose[:, :, 4]))))
    audit.check('independent_jacobian_and_euler_chart_validity',
                sigmas.min() > .001 and sigma_error <= 2e-10 and minimum_cos > .2,
                minimum_jacobian_sigma=float(sigmas.min()), maximum_sigma_discrepancy=sigma_error,
                minimum_absolute_cos_pitch=minimum_cos)
    limits = np.tile(metadata['torque_limits_Nm'], 2)
    raw_ratio = float(np.max(np.abs(arrays['torque_command'])/limits))
    applied_ratio = float(np.max(np.abs(arrays['torque_applied'])/limits))
    clipped = int(np.count_nonzero(np.any(np.abs(arrays['torque_applied']-arrays['torque_command']) > 1e-12, axis=1)))
    audit.check('raw_and_applied_motor_torques_within_limits_without_clipping',
                raw_ratio <= 1+1e-12 and applied_ratio <= 1+1e-12 and clipped == 0
                and maximum_error(np.max(np.abs(model.actuator_ctrlrange), axis=1), limits) <= 1e-12,
                maximum_raw_limit_ratio=raw_ratio, maximum_applied_limit_ratio=applied_ratio,
                clipped_ticks=clipped)
    positions = np.tile(np.asarray(metadata['joint_position_limits_rad']), (2, 1))
    q = arrays['qpos'][:, :12]
    margin = np.minimum(q-positions[:, 0], positions[:, 1]-q)
    speeds = np.tile(metadata['joint_velocity_limits_rad_s'], 2)
    speed_ratio = float(np.max(np.abs(arrays['qvel'][:, :12])/speeds))
    jump_ratio = float(np.max(np.abs(np.diff(q, axis=0))/(intervals[:, None]*speeds)))
    audit.check('joint_positions_velocities_and_state_continuity', margin.min() > 0 and speed_ratio <= 1+1e-12 and jump_ratio <= 1.05,
                minimum_position_margin_rad=float(margin.min()), maximum_velocity_limit_ratio=speed_ratio,
                maximum_state_step_to_speed_limit_ratio=jump_ratio)
    lower_p, upper_p = float(summary['settings']['prior_lower']), float(summary['settings']['prior_upper'])
    estimates_ok = all(np.all((arrays[key] >= lower_p-1e-12) & (arrays[key] <= upper_p+1e-12))
                       for key in ('theta', 'theta_control'))
    audit.check('all_parameter_estimates_and_truth_inside_prior', estimates_ok
                and np.all((truth >= lower_p) & (truth <= upper_p))
                and maximum_error(summary['true_parameters_offline_only'], truth) <= 1e-13,
                minimum_estimate=float(arrays['theta'].min()), maximum_estimate=float(arrays['theta'].max()),
                prior=[lower_p, upper_p], truth_offline_only=truth)
    continuity_error = maximum_error(arrays['theta_control'][1:], arrays['theta'][:-1])
    audit.check('control_then_parameter_update_order', continuity_error <= 1e-13,
                maximum_discrepancy=continuity_error)
    audit.check('no_saved_contacts_or_mujoco_warnings', np.max(arrays['contact_count']) == 0
                and summary['max_contact_count'] == 0 and not summary['mujoco_warnings'],
                maximum_saved_contacts=int(arrays['contact_count'].max()), warnings=summary['mujoco_warnings'])
    track = times >= 2.
    position_rmse = float(np.sqrt(np.mean(np.sum(object_error[track, :3]**2, axis=1)))*1000)
    angle_rmse = float(np.sqrt(np.mean(np.sum(object_error[track, 3:]**2, axis=1)))*180/np.pi)
    audit.check('tracking_rmse_independently_recomputed',
                abs(position_rmse-summary['position_rmse_after_2s_mm']) <= 1e-8
                and abs(angle_rmse-summary['orientation_rmse_after_2s_deg']) <= 1e-8,
                position_rmse_after_2s_mm=position_rmse, angle_rmse_after_2s_deg=angle_rmse)
    audit.check('reported_cumulative_physical_diagnostics', summary['saturated_control_samples'] == 0
                and summary['minimum_joint_limit_margin_rad'] > 0 and summary['max_joint_speed_limit_ratio'] <= 1+1e-12
                and summary['max_weld_residual'] <= 1e-3,
                source='Runner aggregate; independently supplemented by full-tick checks and sampled replays.',
                maximum_weld_residual=summary['max_weld_residual'])
    validate_force_filter(audit, arrays, summary)
    validate_learning(audit, arrays, learned, summary, truth)
    validate_sensors_and_momentum(audit, model, arrays, metadata, truth)
    replay = replay_intervals(model, arrays, dt, physics_dt, samples)
    audit.check('independent_full_physics_short_interval_replay',
                replay['maximum_qpos_error'] <= 1e-7 and replay['maximum_qvel_error'] <= 1e-5
                and replay['maximum_time_error_s'] <= 1e-10 and replay['maximum_contacts'] == 0
                and replay['warning_count'] == 0, **replay,
                integrator=summary.get('integrator', 'implicitfast'))
    if reference is not None:
        refcase = reference if (reference/'trajectory.npz').is_file() else reference/directory.name
        comparison = compare_reference(arrays, refcase)
        if (refcase/'summary.json').is_file():
            old = json.loads((refcase/'summary.json').read_text())
            comparison['reference_data_target_true_error_norms'] = old.get('data_target_error_norms')
            comparison['current_data_target_true_error_norms'] = summary['data_target_error_norms']
            if 'data_target' in old:
                comparison['data_target_difference_norms'] = np.linalg.norm(
                    np.asarray(summary['data_target'])-np.asarray(old['data_target']), axis=1)
        comparison['interpretation'] = 'Numerical sensitivity of this explicitly modified measured-force identifier; no exact physical-robot guarantee.'
        audit.report['step_comparison'] = plain(comparison)
    return audit.finish(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE/'results_exponential')
    parser.add_argument('--case', choices=CASES+('all',), default='all')
    parser.add_argument('--reference', type=Path, default=None)
    parser.add_argument('--duration-s', type=float, default=None)
    parser.add_argument('--replay-samples', type=int, default=101)
    args = parser.parse_args()
    root = args.results.resolve()
    if not root.is_dir():
        parser.error(f'Results root does not exist: {root}')
    reports = []
    for name in (CASES if args.case == 'all' else (args.case,)):
        directory = root/name
        if not directory.is_dir():
            reports.append(dict(case=name, passed=False, failures=['missing_case_directory']))
            print(f'{name}: FAIL missing case directory', flush=True)
            continue
        try:
            report = validate_case(directory, args.replay_samples, args.reference, args.duration_s)
        except Exception as exc:
            audit = Audit(name)
            audit.check('validator_completed', False, exception_type=type(exc).__name__, message=str(exc))
            report = audit.finish(directory)
        reports.append(report)
        message = 'PASS' if report['passed'] else 'FAIL '+', '.join(report['failures'])
        print(f'{name}: {message} ({len(report["checks"])} checks)', flush=True)
    aggregate = dict(passed=all(report['passed'] for report in reports),
                     cases=[dict(case=r['case'], passed=r['passed'], failures=r['failures']) for r in reports],
                     interpretation='Independent sampled full-dynamics validation, with explicit frozen-target bias floors.')
    (root/'validation_summary.json').write_text(json.dumps(plain(aggregate), ensure_ascii=False, indent=2)+'\n')
    return 0 if aggregate['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
