#!/usr/bin/env python3
"""Independent checks for a retrial of the ORIGINAL JAS composite law.

This validator intentionally contains no frozen-target exponential fit. The
original learning law retains -Y S, the scalar gamma historical term, constant
positive diagonal Upsilon, the paper's raw-Gram IE trigger, and its component
projection. Data-target and true-parameter calculations are offline diagnostics.

The raw-schema helpers below are independent of a future runner/learner class.
They support projected explicit Euler and the box-variational-inequality form
of projected implicit Euler. Full rollout-specific checks can be layered onto
these functions once the final retrial runner and reference schema are fixed.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from validate_results import Audit, CASES, HERE, PROJECT, maximum_error, plain
import validate_results as baseline_validation
from validate_identification import configured_model, independent_task_reference


RAW_SCHEMA = {
    't': '[ticks]',
    'theta_control': '[ticks,arms,parameters], estimate used by current control command',
    'theta': '[ticks,arms,parameters], estimate after current learning step',
    'regressor': '[ticks,arms,parameters,6], original tracking regressor Y',
    'surface': '[ticks,arms,6], original sliding variable S',
    'momentum': '[ticks,arms,parameters,6], original local A.T D_b qdot',
    'remainder': '[ticks,arms,parameters,6], original local momentum remainder',
    'known_torque_left': '[ticks,arms,6], A(t).T (held_old_tau + J(t).T fd)',
    'known_torque_right': '[ticks,arms,6], A(t).T (new_tau + J(t).T fd)',
    'filtered_regressor': '[ticks,arms,parameters,6], acceleration-free Y_f',
    'filtered_torque': '[ticks,arms,6], filtered known torque, no measured force',
    'lambda_min': '[ticks,arms], minimum eigenvalue of raw/frozen Gram',
}


def original_reference(times, center, summary):
    # Unlike the earlier engineering runner, the original retrial reference
    # duration is fixed in advance even for a short schema/pilot rollout.
    reference_summary = dict(summary)
    reference_summary['duration_s'] = max(float(summary['duration_s']),
                                        float(summary['settings']['excitation_duration']))
    return independent_task_reference(times, center, reference_summary)


def verify_original_sources(audit, manifest):
    required = {'bare_jrnl_new_sample4.tex', 'simulation/run_original_retrial.py',
                'simulation/run_experiments.py', 'simulation/design_excitation.py',
                'simulation/models/dual_cr7.xml', 'simulation/models/arm_nominal.xml',
                'simulation/models/scene_metadata.json', 'simulation/models/plant_truth.json'}
    for folder in ('jas', 'original_retrial'):
        required.update(str(path.relative_to(PROJECT)) for path in (HERE/folder).glob('*.py'))
    missing, mismatches = sorted(required-set(manifest)), []
    for relative, expected in manifest.items():
        path = PROJECT/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(relative)
    audit.check('source_manifest_sha256', not missing and not mismatches,
                checked_files=len(manifest), missing_required_sources=missing, mismatches=mismatches)
    assets = json.loads((HERE/'models/asset_sha256.json').read_text())
    bad_assets = [name for name, expected in assets.items()
                  if not (HERE/'models'/name).is_file()
                  or hashlib.sha256((HERE/'models'/name).read_bytes()).hexdigest() != expected]
    audit.check('imported_asset_sha256', not bad_assets, checked_files=len(assets), mismatches=bad_assets)


def audit_source_sensing_and_truth(audit):
    forbidden_attributes = {'qacc', 'qfrc_constraint', 'efc_force', 'qM', 'qfrc_bias',
                            'true_theta', 'data_target', 'force_f', 'regressor_f'}
    bad_reads, bad_imports, gain_mutations = [], [], []
    # Explicit trials instantiate jas.learning.CompositeLearner directly;
    # audit the imported original modules as well as the retrial wrappers.
    for name in ('jas/controller.py', 'jas/learning.py',
                 'original_retrial/controller.py', 'original_retrial/learning.py'):
        path = HERE/name
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
                bad_reads.append(dict(file=name, line=node.lineno, attribute=node.attr))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                           else [alias.name for alias in node.names])
                if any('exponential_identification' in module for module in modules):
                    bad_imports.append(dict(file=name, line=node.lineno, modules=modules))
            if isinstance(node, ast.FunctionDef) and node.name == 'advance':
                for child in ast.walk(node):
                    if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                        if any(any(isinstance(part, ast.Attribute) and part.attr in ('gain', 'gamma')
                                   for part in ast.walk(target)) for target in targets):
                            gain_mutations.append(dict(file=name, line=child.lineno))
    runner = ast.parse((HERE/'run_original_retrial.py').read_text())
    function = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == 'run_case')
    main_loop = next(node for node in function.body if isinstance(node, ast.For))
    force_reads = [node.lineno for node in ast.walk(main_loop)
                   if isinstance(node, ast.Attribute) and node.attr == 'qfrc_constraint']
    updates = [node.lineno for node in ast.walk(main_loop)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'update']
    truth_lines = [node.lineno for node in ast.walk(function)
                   if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'plant_truth.json' in node.value]
    diagnostic_order = bool(force_reads and updates and min(force_reads) > max(updates))
    post_rollout_truth = bool(truth_lines and min(truth_lines) > main_loop.end_lineno)
    state_resets = []
    for node in ast.walk(main_loop):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if any(isinstance(part, ast.Attribute) and part.attr in ('qpos', 'qvel')
                       for part in ast.walk(target)):
                    state_resets.append(node.lineno)
    audit.check('original_online_modules_have_no_truth_force_or_acceleration_inputs',
                not bad_reads and not bad_imports and not gain_mutations,
                forbidden_attribute_reads=bad_reads, engineering_imports=bad_imports,
                within_update_gain_mutations=gain_mutations)
    audit.check('runner_truth_is_post_rollout_and_force_reads_are_diagnostics',
                post_rollout_truth and diagnostic_order and not state_resets,
                truth_access_lines=truth_lines, physical_loop_last_line=main_loop.end_lineno,
                reaction_read_lines=force_reads, learning_update_lines=updates,
                in_loop_direct_state_reset_lines=state_resets,
                scope='AST provenance supplemented by independent raw signal and torque reconstruction.')


def original_direction(theta, regressor, surface, gram, p, gamma):
    """Equation (gengxin), before the componentwise boundary projection."""
    return -np.einsum('...pc,...c->...p', regressor, surface)+gamma*(p-np.einsum('...pq,...q->...p', gram, theta))


def project_direction(theta, direction, lower, upper):
    return np.where(((theta >= upper) & (direction > 0))
                    | ((theta <= lower) & (direction < 0)), 0., direction)


def reconstruct_original_filters(times, momentum, remainder, torque_left,
                                 torque_right, rate):
    """Zero-state integration by parts, with causal held-command side limits."""
    yf = np.zeros_like(momentum)
    tauf = np.zeros_like(torque_left)
    z = np.zeros_like(momentum[0])
    source = remainder-rate*momentum
    for tick, dt in enumerate(np.diff(times), start=1):
        decay = np.exp(-rate*dt)
        new_weight = 1+np.expm1(-rate*dt)/(rate*dt)
        old_weight = -np.expm1(-rate*dt)-new_weight
        z = decay*z+old_weight*source[tick-1]+new_weight*source[tick]
        tauf[tick] = (decay*tauf[tick-1]+old_weight*torque_right[tick-1]
                      +new_weight*torque_left[tick])
        yf[tick] = rate*(momentum[tick]-np.exp(-rate*times[tick])*momentum[0])+z
    return yf, tauf


def reconstruct_original_history(times, yf, tauf, window, omega):
    """Original raw-Gram first IE crossing; neither normalization nor gating."""
    ticks, arms, parameters, _ = yf.shape
    active_g = np.zeros((ticks, arms, parameters, parameters))
    active_p = np.zeros((ticks, arms, parameters))
    lambdas = np.zeros((ticks, arms))
    freeze = [None]*arms
    histories = [deque() for _ in range(arms)]
    old_g = np.zeros((arms, parameters, parameters))
    old_p = np.zeros((arms, parameters))
    for tick, dt in enumerate(np.diff(times), start=1):
        current_g = np.stack([yf[tick, arm]@yf[tick, arm].T for arm in range(arms)])
        current_p = np.stack([yf[tick, arm]@tauf[tick, arm] for arm in range(arms)])
        for arm in range(arms):
            if freeze[arm] is not None:
                active_g[tick, arm] = active_g[tick-1, arm]
                active_p[tick, arm] = active_p[tick-1, arm]
                lambdas[tick, arm] = lambdas[tick-1, arm]
                continue
            dg = .5*dt*(current_g[arm]+old_g[arm])
            dp = .5*dt*(current_p[arm]+old_p[arm])
            histories[arm].append((times[tick], dg, dp))
            g = active_g[tick-1, arm]+dg
            p = active_p[tick-1, arm]+dp
            while histories[arm] and histories[arm][0][0] <= times[tick]-window:
                _, first_g, first_p = histories[arm].popleft()
                g -= first_g; p -= first_p
            g = .5*(g+g.T)
            active_g[tick, arm], active_p[tick, arm] = g, p
            lambdas[tick, arm] = np.linalg.eigvalsh(g)[0]
            if times[tick] >= window and lambdas[tick, arm] >= omega:
                freeze[arm] = float(times[tick])
        old_g, old_p = current_g, current_p
    return active_g, active_p, lambdas, freeze


def audit_original_filter_and_history(audit, arrays, learned, settings, reported_freeze):
    fields = ('t', 'momentum', 'remainder', 'known_torque_left', 'known_torque_right',
              'filtered_regressor', 'filtered_torque', 'lambda_min')
    missing = [name for name in fields if name not in arrays]
    audit.check('original_local_filter_raw_channels_present', not missing, missing=missing)
    if missing:
        return None
    yf, tauf = reconstruct_original_filters(arrays['t'], arrays['momentum'], arrays['remainder'],
                                           arrays['known_torque_left'], arrays['known_torque_right'],
                                           float(settings['filter_rate']))
    yerr, terr = maximum_error(yf, arrays['filtered_regressor']), maximum_error(tauf, arrays['filtered_torque'])
    audit.check('original_local_momentum_filter_and_causal_known_torque', yerr <= 2e-8 and terr <= 2e-8,
                maximum_regressor_error=yerr, maximum_torque_error=terr,
                interpretation='Original local six-dimensional filter only; no measured reaction or payload observation block.')
    # After independently checking the filter, use its recorded raw outputs
    # for the history/update audit. This avoids amplifying harmless filter
    # reconstruction roundoff by a deliberately stiff gamma and Upsilon.
    g, p, lambdas, freeze = reconstruct_original_history(arrays['t'], arrays['filtered_regressor'], arrays['filtered_torque'],
                                                         float(settings['window']), float(settings['omega']))
    detected_ok = len(freeze) == len(reported_freeze) and all(
        (x is None and y is None) or (x is not None and y is not None and abs(x-y) <= 1e-10)
        for x, y in zip(freeze, reported_freeze))
    audit.check('original_raw_gram_ie_first_threshold_crossing', detected_ok,
                reconstructed_times_s=freeze, reported_times_s=reported_freeze,
                window_s=settings['window'], threshold=settings['omega'],
                trigger='lambda_min(raw Gram)>=omega, first local crossing at or after window; no target or normalized threshold.')
    ge, pe = maximum_error(g[-1], learned['gram']), maximum_error(p[-1], learned['p'])
    gt = 5e-8+2e-13*float(np.max(np.abs(g[-1])))
    pt = 5e-8+2e-13*float(np.max(np.abs(p[-1])))
    audit.check('original_frozen_gram_and_p_rebuilt_from_raw_samples', ge <= gt and pe <= pt,
                maximum_gram_error=ge, gram_tolerance=gt, maximum_p_error=pe, p_tolerance=pt)
    le = maximum_error(lambdas, arrays['lambda_min'])
    audit.check('original_raw_minimum_eigenvalue_log', le <= 2e-7, maximum_error=le)
    return g, p


def audit_original_learning_update(audit, arrays, gram, p, gain, gamma,
                                   lower=.85, upper=1.15, method='projected_euler'):
    """Every tick: equation (gengxin) and either explicit or implicit projection.

    For implicit Euler the check is the box KKT/fixed point, not solving the
    unconstrained linear system and clipping its answer. This distinction
    matters because a non-diagonal history Gram couples active coordinates.
    """
    theta, old = arrays['theta'], arrays['theta_control']
    gains = np.asarray(gain, dtype=float)
    if gains.ndim == 0:
        gains = np.full(theta.shape[1:], float(gains))
    elif gains.shape == (theta.shape[2],):
        gains = np.broadcast_to(gains, theta.shape[1:])
    gamma_value = np.asarray(gamma, dtype=float)
    gamma_ok = gamma_value.ndim <= 1 and np.all(np.isfinite(gamma_value)) and np.all(gamma_value > 0)
    gains_ok = (gains.shape == theta.shape[1:] and np.all(np.isfinite(gains))
                and np.all(gains > 0) and gamma_ok)
    audit.check('constant_positive_diagonal_original_learning_gain', gains_ok,
                constant_gain_diagonal=gains, gamma=gamma)
    if not gains_ok:
        return
    dt = np.diff(arrays['t'])[:, None, None]
    rg, surface = arrays['regressor'], arrays['surface']
    gamma_array = np.asarray(gamma, dtype=float)
    if gamma_array.ndim == 1:
        gamma_array = gamma_array[None, :, None]
    continuity = maximum_error(old[1:], theta[:-1])
    audit.check('original_control_then_learning_update_order', continuity <= 1e-13,
                maximum_error=continuity)
    prior_ok = np.all((theta >= lower-1e-12) & (theta <= upper+1e-12))
    audit.check('original_componentwise_prior_invariance', prior_ok,
                minimum_estimate=float(theta.min()), maximum_estimate=float(theta.max()), lower=lower, upper=upper)
    epsilon_old = p-np.einsum('napq,naq->nap', gram, old)
    tracking = -np.einsum('napc,nac->nap', rg, surface)
    gamma_times_old = gamma_array*epsilon_old
    if 'prediction_error' in arrays:
        # The runner must declare which estimate its diagnostic epsilon uses.
        epsilon_logged = epsilon_old
        if arrays.get('_epsilon_uses_post_update', False):
            epsilon_logged = p-np.einsum('napq,naq->nap', gram, theta)
        error = maximum_error(epsilon_logged, arrays['prediction_error'])
        audit.check('original_prediction_error_contains_raw_history_residual', error <= 2e-6,
                    maximum_error=error)
    if method in ('projected_euler', 'explicit_euler'):
        direction = project_direction(old[1:], tracking[1:]+gamma_times_old[1:], lower, upper)
        expected = np.clip(old[1:]+dt*gains*direction, lower, upper)
        error = maximum_error(expected, theta[1:])
        audit.check('every_tick_original_projected_explicit_learning_law', error <= 3e-10,
                    maximum_parameter_step_error=error, tolerance=3e-10,
                    retained_terms='-Y S plus gamma*(p-Gram theta), multiplied by constant diagonal Upsilon')
    elif method in ('projected_implicit', 'projected_implicit_euler', 'implicit_euler', 'box_implicit_euler'):
        epsilon_new = p-np.einsum('napq,naq->nap', gram, theta)
        if 'new_prediction_error' in arrays:
            error = maximum_error(epsilon_new, arrays['new_prediction_error'])
            audit.check('implicit_prediction_error_uses_updated_parameter_only_in_numerical_solve',
                        error <= 2e-6, maximum_error=error)
        force_new = tracking+gamma_array*epsilon_new
        fixed_point = np.clip(old[1:]+dt*gains*force_new[1:], lower, upper)
        fp_error = maximum_error(theta[1:], fixed_point)
        residual = (theta[1:]-old[1:])/(dt*gains)-force_new[1:]
        at_lower, at_upper = theta[1:] <= lower+1e-10, theta[1:] >= upper-1e-10
        violation = np.where(at_lower, np.minimum(residual, 0.),
                             np.where(at_upper, np.maximum(residual, 0.), residual))
        scale = 1+np.abs(force_new[1:])+np.abs((theta[1:]-old[1:])/(dt*gains))
        relative = float(np.max(np.abs(violation)/scale))
        audit.check('every_tick_original_projected_implicit_learning_box_vi',
                    fp_error <= 2e-7 and relative <= 2e-7,
                    maximum_parameter_fixed_point_error=fp_error,
                    maximum_scaled_kkt_violation=relative,
                    maximum_absolute_kkt_violation=float(np.max(np.abs(violation))),
                    retained_terms='-Y S plus gamma*(p-Gram theta_new), constant diagonal Upsilon; box KKT projection',
                    interpretation='Backward Euler approximation of the original continuous projected law; step sensitivity is required.')
    else:
        audit.check('recognized_original_learning_integrator', False, method=method)


def audit_offline_residual(audit, learned, truth):
    residual = learned['p']-np.einsum('apq,aq->ap', learned['gram'], truth)
    if 'stored_residual' in learned:
        error = maximum_error(residual, learned['stored_residual'])
        audit.check('offline_original_stored_residual_not_removed_or_compensated', error <= 2e-6,
                    maximum_error=error, residual_norms=np.linalg.norm(residual, axis=1),
                    interpretation='This is offline only. Nonzero history residual precludes an unconditional exact-truth exponential claim.')
    return residual


def compare_original_learning_steps(arrays, learned, reference):
    """Disclose learning-step sensitivity separately from pose sensitivity.

    A sign controller can change the sampled data when the step changes.  This
    is a numerical comparison, not a truth-convergence acceptance condition.
    """
    with np.load(reference/'trajectory.npz', allow_pickle=False) as source:
        reference_t = source['t']
        reference_theta = source['theta']
    with np.load(reference/'learning_state.npz', allow_pickle=False) as source:
        reference_gram, reference_p = source['gram'], source['p']
    valid = (arrays['t'] >= reference_t[0]) & (arrays['t'] <= reference_t[-1])
    times = arrays['t'][valid]
    interpolated = np.empty_like(arrays['theta'][valid])
    for arm in range(interpolated.shape[1]):
        for parameter in range(interpolated.shape[2]):
            interpolated[:, arm, parameter] = np.interp(
                times, reference_t, reference_theta[:, arm, parameter])
    difference = arrays['theta'][valid]-interpolated
    summary = json.loads((reference/'summary.json').read_text())
    return dict(
        reference_directory=str(reference.resolve()), common_samples=len(times),
        maximum_absolute_parameter_trace_difference=np.max(np.abs(difference), axis=0),
        rms_parameter_trace_difference=np.sqrt(np.mean(difference**2, axis=0)),
        final_common_parameter_difference=difference[-1],
        frozen_gram_relative_difference=np.linalg.norm(learned['gram']-reference_gram, axis=(1, 2))
            /(1+np.linalg.norm(reference_gram, axis=(1, 2))),
        frozen_p_relative_difference=np.linalg.norm(learned['p']-reference_p, axis=1)
            /(1+np.linalg.norm(reference_p, axis=1)),
        reference_ie_detection_times_s=summary['ie_detection_times_s'],
        interpretation='Sampled original-law sensitivity; this does not fit an exponential or certify a zero true-parameter error.')


def audit_original_control_signals(audit, arrays, summary, samples=41):
    """Rebuild local paper formulas without calling the retrial controller."""
    from types import SimpleNamespace
    from jas.barrier import transform, terminal_surface
    from jas.dynamics import LinkDynamics, coupled_bases
    from jas.reference import desired, envelope
    from design_excitation import multisine_reference
    from run_experiments import kinematics
    metadata = json.loads((HERE/'models/scene_metadata.json').read_text())
    c = SimpleNamespace(**summary['settings'])
    center = np.r_[metadata['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    payload = metadata['payload_nominal']
    nominal = LinkDynamics(HERE/'models/arm_nominal.xml')
    kins = [kinematics(metadata, name) for name in ('left', 'right')]
    indices = np.unique(np.linspace(0, len(arrays['t'])-1, min(samples, len(arrays['t']))).astype(int))
    maxima = {key: 0. for key in ('torque', 'regressor', 'momentum', 'remainder', 'surface',
                                 'known_torque_left', 'known_torque_right', 'learning_drive')}
    for tick in indices:
        t = float(arrays['t'][tick])
        reference = (desired(t, center) if c.excitation_profile == 'baseline' else
                     multisine_reference(t, center, c.excitation_duration, c.excitation_profile))
        ref, dref, ddref = reference
        for arm in range(2):
            s = slice(6*arm, 6*arm+6)
            q, qd = arrays['qpos'][tick, s], arrays['qvel'][tick, s]
            pose, jacobian, jacobian_dot = kins[arm].evaluate(q, qd)
            bs = transform(pose-ref, jacobian@qd-dref,
                           envelope(t, np.asarray(c.chi), np.asarray(c.ell), summary['scenario']['nonmonotonic']),
                           np.asarray(c.chi), np.asarray(c.ell))
            surface, xir, xir_dot = terminal_surface(bs.xi, bs.xidot, c.lam1, c.lam2, c.alpha, c.beta, c.delta)
            inverse_j = np.linalg.inv(jacobian)
            a = inverse_j/bs.a
            a_dot = -inverse_j@jacobian_dot@a-a*(bs.adot/bs.a)
            velocity_known = bs.a*dref-bs.b
            v = a@velocity_known
            vdot = a_dot@velocity_known+a@(bs.adot*dref+bs.a*ddref-bs.bdot)
            q_reference_velocity = a@xir+v
            q_reference_acceleration = a_dot@xir+a@xir_dot+vdot
            robot = nominal.evaluate(q, qd)
            db, cb, wb = coupled_bases(robot, jacobian, jacobian_dot, pose, jacobian@qd,
                                       payload['mass_kg'], payload['diagonal_inertia_kg_m2'])
            wb[6] = .05*qd
            y = np.stack([a.T@(db[p]@q_reference_acceleration+cb[p]@q_reference_velocity+wb[p]) for p in range(8)])
            momentum = np.stack([a.T@db[p]@qd for p in range(8)])
            remainder = np.stack([a.T@(wb[p]-cb[p].T@qd)-a_dot.T@db[p]@qd for p in range(8)])
            force_desired = np.array([0., (-1 if arm == 0 else 1)*c.internal_force, 0., 0., 0., 0.])
            uncertainty_width = c.prior_upper-c.prior_lower
            robust = y.T@(uncertainty_width*np.sign(y@surface))
            rho = c.rho_hat*np.linalg.norm(a, ord=2)**2
            feedback = (y.T@arrays['theta_control'][tick, arm]-(np.asarray(c.ka)+rho)*surface
                        -np.asarray(c.k2)*np.sign(surface)*np.abs(surface)**c.upsilon1
                        -np.asarray(c.k3)*np.sign(surface)*np.abs(surface)**c.upsilon2-robust)
            torque = jacobian.T@(bs.a*feedback-force_desired)
            held = arrays['torque_applied'][tick, s] if tick == 0 else arrays['torque_applied'][tick-1, s]
            known_left = a.T@(held+jacobian.T@force_desired)
            known_right = a.T@(arrays['torque_applied'][tick, s]+jacobian.T@force_desired)
            reconstructed = dict(torque=torque, regressor=y, momentum=momentum, remainder=remainder,
                                 surface=surface, known_torque_left=known_left, known_torque_right=known_right,
                                 learning_drive=y@surface)
            for name, value in reconstructed.items():
                key = 'torque_command' if name == 'torque' else name
                target = arrays[key][tick, s] if name == 'torque' else arrays[key][tick, arm]
                maxima[name] = max(maxima[name], maximum_error(value, target))
    audit.check('original_full_sign_torque_formula_reconstructed_from_local_encoders',
                maxima['torque'] <= 2e-8, sampled_ticks=len(indices), arm_samples=2*len(indices),
                maximum_torque_discrepancy_Nm=maxima['torque'],
                retained='Y.T theta_control, positive rho, Ka/K2/K3, original 8D sign(YS), prior width, desired fd; no actual force.')
    audit.check('original_tracking_regressor_surface_and_YS_reconstructed',
                max(maxima['regressor'], maxima['surface'], maxima['learning_drive']) <= 2e-8,
                maximum_regressor_discrepancy=maxima['regressor'],
                maximum_surface_discrepancy=maxima['surface'], maximum_YS_discrepancy=maxima['learning_drive'])
    audit.check('original_local_momentum_remainder_and_known_torque_reconstructed',
                max(maxima[name] for name in ('momentum', 'remainder', 'known_torque_left', 'known_torque_right')) <= 2e-8,
                maximum_signal_discrepancies={name: maxima[name] for name in ('momentum', 'remainder', 'known_torque_left', 'known_torque_right')},
                sensing='Original six-dimensional locally derived signals; no reaction-force or actual payload sensor inputs.')


def audit_physical_force_and_local_model(audit, arrays, summary, truth, samples=32):
    """Offline force/basis cross-check, explicitly outside both online laws."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from validate_identification import NominalNewtonEuler
    from jas.dynamics import LinkDynamics, coupled_bases
    from jas.geometry import angular_map
    from run_experiments import kinematics
    metadata = json.loads((HERE/'models/scene_metadata.json').read_text())
    payload = metadata['payload_nominal']
    mass = float(payload['mass_kg']); inertia = np.diag(payload['diagonal_inertia_kg_m2'])
    model = configured_model(summary)
    nominal = NominalNewtonEuler(); nominal_matrices = LinkDynamics(HERE/'models/arm_nominal.xml')
    kins = [kinematics(metadata, name) for name in ('left', 'right')]
    indices = np.unique(np.linspace(0, len(arrays['t'])-1, min(samples, len(arrays['t']))).astype(int))
    robot_error = payload_error = parameter_basis_error = force_rows_error = 0.
    independent_internal_error = 0.
    acceleration_gaps, angular_acceleration_gaps = [], []
    for tick in indices:
        data = mujoco.MjData(model)
        data.qpos[:] = arrays['qpos'][tick]; data.qvel[:] = arrays['qvel'][tick]
        data.ctrl[:] = arrays['torque_applied'][tick]; data.time = arrays['t'][tick]
        mujoco.mj_forward(model, data)
        # An independent projection of the actual equality multipliers checks
        # that no contact/limit contribution or force sign was misattributed.
        eq_force = np.asarray(data.efc_force).copy()
        eq_force[np.asarray(data.efc_type) != mujoco.mjtConstraint.mjCNSTR_EQUALITY] = 0.
        equality_qforce = np.zeros(model.nv)
        mujoco.mj_mulJacTVec(model, data, equality_qforce, eq_force)
        force_rows_error = max(force_rows_error, maximum_error(equality_qforce, data.qfrc_constraint))
        payload_body = model.body('payload').id
        actual_rotation = data.xmat[payload_body].reshape(3, 3)
        actual_omega = actual_rotation@data.qvel[15:18]
        actual_alpha = actual_rotation@data.qacc[15:18]
        actual_inertia = actual_rotation@inertia@actual_rotation.T
        actual_wrench = np.r_[data.qfrc_constraint[12:15], actual_rotation@data.qfrc_constraint[15:18]]
        payload_prediction = truth[0, 7]*np.r_[mass*(data.qacc[12:15]-model.opt.gravity),
                          actual_inertia@actual_alpha+np.cross(actual_omega, actual_inertia@actual_omega)]
        payload_error = max(payload_error, maximum_error(actual_wrench, payload_prediction))
        arm_forces = []
        for arm in range(2):
            s = slice(6*arm, 6*arm+6)
            q, qd, qdd = data.qpos[s], data.qvel[s], data.qacc[s]
            robot_basis = nominal.forces(q, qd, qdd)
            robot_prediction = truth[arm, :6]@robot_basis+truth[arm, 6]*.05*qd
            robot_error = max(robot_error, maximum_error(robot_prediction,
                                                         data.qfrc_actuator[s]+data.qfrc_constraint[s]))
            pose, j, jdot = kins[arm].evaluate(q, qd)
            xd, xdd = j@qd, j@qdd+jdot@qd
            mapping = angular_map(pose[3:])
            _, pitch, yaw = pose[3:]
            cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
            derivative_pitch = np.array([[-cy*sp, 0., 0.], [-sy*sp, 0., 0.], [-cp, 0., 0.]])
            derivative_yaw = np.array([[-sy*cp, -cy, 0.], [cy*cp, -sy, 0.], [0., 0., 0.]])
            mapping_dot = derivative_pitch*xd[4]+derivative_yaw*xd[5]
            omega = mapping@xd[3:]
            alpha = mapping@xdd[3:]+mapping_dot@xd[3:]
            rotation = Rotation.from_euler('xyz', pose[3:]).as_matrix()
            world_inertia = rotation@inertia@rotation.T
            ideal_payload_basis = np.r_[mass*(xdd[:3]-model.opt.gravity),
                    mapping.T@(world_inertia@alpha+np.cross(omega, world_inertia@omega))]
            db, cb, wb = coupled_bases(nominal_matrices.evaluate(q, qd), j, jdot, pose, xd,
                                       mass, payload['diagonal_inertia_kg_m2'])
            wb[6] = .05*qd
            generalized = np.stack([db[p]@qdd+cb[p]@qd+wb[p] for p in range(8)])
            independent_basis = np.vstack((robot_basis, .05*qd, .5*j.T@ideal_payload_basis))
            parameter_basis_error = max(parameter_basis_error, maximum_error(generalized, independent_basis))
            # A small weld pose error is not evidence of equal acceleration.
            acceleration_gaps.append(float(np.linalg.norm(xdd[:3]-data.qacc[12:15])))
            angular_acceleration_gaps.append(float(np.linalg.norm(alpha-actual_alpha)))
            arm_forces.append(np.linalg.solve(j.T, data.qfrc_constraint[s]))
        arm_forces = np.array(arm_forces)
        internal = arm_forces-.5*arm_forces.sum(axis=0)
        independent_internal_error = max(independent_internal_error,
                                         maximum_error(internal, arrays['internal_force'][tick]))
    audit.check('offline_true_robot_and_payload_newton_euler_force_balances',
                robot_error <= 2e-6 and payload_error <= 2e-6 and force_rows_error <= 2e-7,
                sampled_ticks=len(indices), maximum_robot_torque_residual_Nm=robot_error,
                maximum_payload_wrench_component_residual=payload_error,
                maximum_constraint_multiplier_projection_error=force_rows_error,
                scope='Validation-only qacc and truth; they are never supplied to either online law.')
    audit.check('original_coupled_inertia_regressor_matches_independent_local_newton_euler',
                parameter_basis_error <= 2e-6 and independent_internal_error <= 2e-7,
                maximum_local_eight_parameter_basis_error=parameter_basis_error,
                maximum_saved_internal_force_discrepancy=independent_internal_error,
                maximum_local_vs_actual_payload_linear_acceleration_gap_m_s2=max(acceleration_gaps),
                rms_local_vs_actual_payload_linear_acceleration_gap_m_s2=float(np.sqrt(np.mean(np.square(acceleration_gaps)))),
                maximum_local_vs_actual_payload_angular_acceleration_gap_rad_s2=max(angular_acceleration_gaps),
                interpretation='The nominal basis is checked independently. Finite-weld acceleration mismatch remains a model residual, not a corrected online measurement.')


def run_legacy_physics_checks(directory, summary, samples, reference, expected_duration):
    """Reuse read-only old checks with process-local, restored helper bindings."""
    original_reference_function = baseline_validation.independent_reference
    original_sources_function = baseline_validation.verify_sources
    original_replay_function = baseline_validation.replay_intervals
    def reference_helper(times, center):
        return original_reference(times, center, summary)
    def replay_helper(model, arrays, control_dt, physics_dt, count):
        model.opt.integrator = configured_model(summary).opt.integrator
        return original_replay_function(model, arrays, control_dt, physics_dt, count)
    baseline_validation.independent_reference = reference_helper
    baseline_validation.verify_sources = verify_original_sources
    baseline_validation.replay_intervals = replay_helper
    try:
        duration = float(summary['duration_s']) if expected_duration is None else expected_duration
        return baseline_validation.validate_case(directory, duration, samples, reference)
    finally:
        baseline_validation.independent_reference = original_reference_function
        baseline_validation.verify_sources = original_sources_function
        baseline_validation.replay_intervals = original_replay_function


def validate_case(directory, samples=101, reference=None, expected_duration=None):
    summary = json.loads((directory/'summary.json').read_text())
    report = run_legacy_physics_checks(directory, summary, samples, reference, expected_duration)
    audit = Audit(directory.name)
    audit.report = report
    audit_source_sensing_and_truth(audit)
    with np.load(directory/'trajectory.npz', allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    with np.load(directory/'learning_state.npz', allow_pickle=False) as source:
        learned = {key: source[key] for key in source.files}
    settings = summary['settings']
    audit.check('original_sign_feedback_declared', settings.get('robust_mode') == 'paper_sign',
                robust_mode=settings.get('robust_mode'))
    history = audit_original_filter_and_history(audit, arrays, learned, settings,
                                                 summary['ie_detection_times_s'])
    if history is not None:
        missing = [key for key in ('regressor', 'surface', 'theta_control', 'theta') if key not in arrays]
        audit.check('original_update_raw_channels_present', not missing, missing=missing)
        if not missing:
            gain = settings.get('adaptation_gain_diagonal', settings.get('adaptation_gain'))
            method = summary.get('learning_integrator', settings.get('learning_integrator', 'projected_euler'))
            arrays['_epsilon_uses_post_update'] = bool(summary.get('epsilon_uses_post_update', False))
            audit_original_learning_update(audit, arrays, *history, gain, settings['gamma'],
                                           settings['prior_lower'], settings['prior_upper'], method)
            required = ('qpos', 'qvel', 'torque_command', 'torque_applied', 'learning_drive')
            absent = [name for name in required if name not in arrays]
            audit.check('original_control_reconstruction_raw_channels_present', not absent, missing=absent)
            if not absent:
                audit_original_control_signals(audit, arrays, summary)
    truth_record = json.loads((HERE/'models/plant_truth.json').read_text())
    truth = np.array([truth_record[name+'_density_multipliers']+
                      [truth_record[name+'_viscous_multiplier'], truth_record['payload_density_multiplier']]
                      for name in ('left', 'right')])
    audit_offline_residual(audit, learned, truth)
    audit_physical_force_and_local_model(audit, arrays, summary, truth)
    if reference is not None:
        reference_case = reference if (reference/'trajectory.npz').is_file() else reference/directory.name
        audit.report['original_learning_step_comparison'] = compare_original_learning_steps(
            arrays, learned, reference_case)
    audit.report['scope'] = ('Original continuous JAS sign control and full projected composite learning; '
                              'all-tick raw filter/history/update, full-scene FK/barriers/limits, '
                              'sampled local torque/regression reconstruction and full-physics replays. '
                              'Nonzero local history residual is reported without claiming exact truth convergence.')
    return audit.finish(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--case', choices=CASES+('all',), default='all')
    parser.add_argument('--duration-s', type=float, default=None)
    parser.add_argument('--reference', type=Path, default=None)
    parser.add_argument('--replay-samples', type=int, default=101)
    args = parser.parse_args()
    reports = []
    for name in (CASES if args.case == 'all' else (args.case,)):
        directory = args.results/name
        if not directory.is_dir():
            reports.append(dict(case=name, passed=False, failures=['missing_case_directory']))
            print(f'{name}: FAIL missing_case_directory', flush=True)
            continue
        try:
            report = validate_case(directory, args.replay_samples, args.reference, args.duration_s)
        except Exception as exc:
            audit = Audit(name)
            audit.check('validator_completed', False, exception_type=type(exc).__name__, message=str(exc))
            report = audit.finish(directory)
        reports.append(report)
        print(f'{name}: {"PASS" if report["passed"] else "FAIL " + ", ".join(report["failures"])}', flush=True)
    aggregate = dict(passed=all(report['passed'] for report in reports),
                     cases=[dict(case=r['case'], passed=r['passed'], failures=r['failures']) for r in reports])
    (args.results/'validation_summary.json').write_text(json.dumps(plain(aggregate), ensure_ascii=False, indent=2)+'\n')
    return 0 if aggregate['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
