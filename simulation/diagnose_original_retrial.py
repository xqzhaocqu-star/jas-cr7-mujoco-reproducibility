#!/usr/bin/env python3
"""Offline internal-force decomposition of ORIGINAL JAS historical residuals.

Raw trajectories, learning records, models and controllers are read only.
The only generated artifact is CASE/residual_diagnosis.json. Replayed
instantaneous constraint reactions never enter control or learning, and no
physical trajectory is re-integrated or replaced. The remaining residual
includes finite-weld/chart, sampling/filter, and numerical approximations;
its norm alone does not predict the full original closed-loop limit.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from run_experiments import kinematics


HERE = Path(__file__).resolve().parent
REQUIRED = ('t', 'qpos', 'qvel', 'torque_applied', 'filtered_regressor',
            'filtered_torque', 'local_error', 'upper', 'lower',
            'internal_force', 'interaction_force', 'desired_internal_force',
            'contact_count')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _first_order_hold(state, previous_right, current_left, dt, rate):
    """Exact zero-state low-pass propagation of affine interval signals."""
    x = rate*dt
    exponential_minus_one = np.expm1(-x)
    return (np.exp(-x)*state-exponential_minus_one*previous_right
            +(1+exponential_minus_one/x)*(current_left-previous_right))


def _sources(directory):
    manifest = json.loads((directory/'source_manifest.json').read_text())
    required = ('simulation/models/dual_cr7.xml', 'simulation/models/arm_nominal.xml',
                'simulation/models/scene_metadata.json', 'simulation/jas/geometry.py',
                'simulation/run_experiments.py')
    verified = {}
    for name in required:
        actual = _sha256(HERE.parent/name)
        expected = manifest.get(name)
        if expected is None or actual != expected:
            raise ValueError(f'{directory}: model/kinematic source differs from its recorded manifest: {name}')
        verified[name] = actual
    return verified


def _barrier_derivative(arrays, settings):
    """Independent d(eta/[(u-eta)(v+eta)])/de from recorded errors/bounds."""
    chi, ell = np.asarray(settings['chi']), np.asarray(settings['ell'])
    u = ell*np.tanh(chi*arrays['upper'])
    v = -ell*np.tanh(chi*arrays['lower'])
    stretched = np.tanh(chi*arrays['local_error'])
    eta = ell*stretched
    gap_upper, gap_lower = u[:, None, :]-eta, v[:, None, :]+eta
    if np.any(gap_upper <= 0) or np.any(gap_lower <= 0):
        raise ValueError('Recorded local errors leave the original barrier domain.')
    derivative = (chi*ell*(1-stretched*stretched)
                  *(u[:, None, :]*v[:, None, :]+eta*eta)/(gap_upper*gap_lower)**2)
    if not np.all(np.isfinite(derivative)) or np.any(derivative <= 0):
        raise ValueError('Nonpositive or nonfinite barrier derivative in offline reconstruction.')
    return derivative


def _integral(integrands, times, end, window):
    """Match original saved-grid sliding intervals, including rounding edges."""
    indices = np.flatnonzero(times[1:end+1] > times[end]-window)+1
    if not len(indices):
        return np.zeros_like(integrands[0]), indices
    weights = .5*(times[indices]-times[indices-1])
    shape = (len(weights),)+(1,)*(integrands.ndim-1)
    value = np.sum(weights.reshape(shape)*(integrands[indices]+integrands[indices-1]), axis=0)
    return value, indices


def _first_ie_from_raw(times, gram_integrands, window, threshold):
    dt = np.diff(times)
    increments = .5*dt[:, None, None, None]*(gram_integrands[1:]+gram_integrands[:-1])
    cumulative = np.concatenate((np.zeros_like(gram_integrands[:1]),
                                 np.cumsum(increments, axis=0)))
    excluded = np.maximum(np.searchsorted(times, times-window, side='right')-1, 0)
    rolling = cumulative-cumulative[excluded]
    minimum = np.linalg.eigvalsh(rolling)[..., 0]
    detected = []
    for arm in range(2):
        accepted = np.flatnonzero((times >= window) & (minimum[:, arm] >= threshold))
        detected.append(float(times[accepted[0]]) if len(accepted) else None)
    return detected


def diagnose_case(directory):
    directory = Path(directory).resolve()
    summary = json.loads((directory/'summary.json').read_text())
    settings = summary['settings']
    if settings.get('robust_mode') != 'paper_sign' or 'identification_rate' in settings:
        raise ValueError('This diagnostic is restricted to the original JAS learning retrial.')
    verified_sources = _sources(directory)
    with np.load(directory/'trajectory.npz', allow_pickle=False) as saved:
        missing = [name for name in REQUIRED if name not in saved.files]
        if missing:
            raise ValueError(f'Missing original recorded arrays: {missing}')
        arrays = {name: saved[name] for name in REQUIRED}
    with np.load(directory/'learning_state.npz', allow_pickle=False) as saved:
        gram, p, truth = (saved[name] for name in ('gram', 'p', 'true_parameters_offline_only'))
        stored_residual = saved['stored_residual']
    times = arrays['t']
    if abs(times[0]) > 1e-12 or not np.all(np.diff(times) > 0):
        raise ValueError('The zero-state filter requires the full strictly increasing trajectory from t=0.')
    if int(np.max(arrays['contact_count'])) != 0 or summary.get('max_contact_count', 0) != 0:
        raise ValueError('Other contacts contaminate the interpretation of qfrc_constraint as grasp reactions.')
    if summary.get('minimum_joint_limit_margin_rad', 0.) <= 0:
        raise ValueError('Joint limit reactions invalidate the grasp-only force decomposition.')
    if truth.shape != (2, 8) or arrays['filtered_regressor'].shape != (len(times), 2, 8, 6):
        raise ValueError('Expected the declared two-arm eight-parameter, six-channel regression.')
    for name in REQUIRED:
        if name not in ('upper', 'lower') and not np.all(np.isfinite(arrays[name])):
            raise ValueError(f'Nonfinite recorded data: {name}')

    yf, tau_f = arrays['filtered_regressor'], arrays['filtered_torque']
    gram_integrands = np.einsum('naik,najk->naij', yf, yf)
    p_integrands = np.einsum('naik,nak->nai', yf, tau_f)
    predicted = np.einsum('naik,ai->nak', yf, truth)
    total_filtered_residual = tau_f-predicted
    total_integrands = np.einsum('naik,nak->nai', yf, total_filtered_residual)
    window, threshold = float(settings['window']), float(settings['omega'])
    first_ie = _first_ie_from_raw(times, gram_integrands, window, threshold)
    reported_ie = summary['ie_detection_times_s']
    for actual, reported in zip(first_ie, reported_ie):
        if (actual is None) != (reported is None) or (actual is not None and abs(actual-reported) > 2e-10):
            raise ValueError(f'Original first IE differs from raw reintegration: {first_ie} vs {reported_ie}')
    ends = [len(times)-1 if te is None else int(np.argmin(np.abs(times-te))) for te in reported_ie]
    last = max(ends)+1
    barrier_derivative = _barrier_derivative(arrays, settings)
    desired = arrays['desired_internal_force']
    if np.max(np.abs(np.sum(desired, axis=1))) > 1e-12:
        raise ValueError('Desired internal forces do not satisfy the original force-balance condition.')
    right_d = (arrays['internal_force']-desired)/barrier_derivative
    filtered_d = np.zeros((last, 2, 6))

    # The plant model is used only as an offline reaction oracle. No mj_step
    # or qacc access occurs here. Each replay evaluates the RECORDED state.
    metadata = json.loads((HERE/'models'/'scene_metadata.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(HERE/'models'/'dual_cr7.xml'))
    model.opt.timestep = float(summary['physics_dt_s'])
    if summary.get('integrator') in ('RK4', 'implicitfast'):
        model.opt.integrator = {'RK4': mujoco.mjtIntegrator.mjINT_RK4,
                                'implicitfast': mujoco.mjtIntegrator.mjINT_IMPLICITFAST}[summary['integrator']]
    data = mujoco.MjData(model)
    local = [kinematics(metadata, name) for name in ('left', 'right')]
    maximum_right_force_error = 0.
    maximum_right_internal_force_error = 0.
    maximum_left_right_jump = 0.
    rate = float(settings['filter_rate'])
    for n in range(last):
        data.qpos[:] = arrays['qpos'][n]
        data.qvel[:] = arrays['qvel'][n]
        data.time = float(times[n])
        jacobians = [local[arm].pose_jacobian(arrays['qpos'][n, 6*arm:6*arm+6])[1]
                     for arm in range(2)]

        def reactions(applied):
            data.ctrl[:] = applied
            data.qacc_warmstart[:] = 0.
            mujoco.mj_forward(model, data)
            return np.array([np.linalg.solve(jacobians[arm].T,
                                             data.qfrc_constraint[6*arm:6*arm+6])
                             for arm in range(2)])

        right_force = reactions(arrays['torque_applied'][n])
        maximum_right_force_error = max(maximum_right_force_error,
                                        float(np.max(np.abs(right_force-arrays['interaction_force'][n]))))
        right_internal = right_force-.5*np.sum(right_force, axis=0)
        maximum_right_internal_force_error = max(maximum_right_internal_force_error,
            float(np.max(np.abs(right_internal-arrays['internal_force'][n]))))
        if n == 0:
            continue
        left_force = reactions(arrays['torque_applied'][n-1])
        left_internal = left_force-.5*np.sum(left_force, axis=0)
        left_d = (left_internal-desired[n])/barrier_derivative[n]
        maximum_left_right_jump = max(maximum_left_right_jump,
                                     float(np.max(np.abs(left_force-right_force))))
        filtered_d[n] = _first_order_hold(filtered_d[n-1], right_d[n-1], left_d,
                                           times[n]-times[n-1], rate)
    if max(maximum_right_force_error, maximum_right_internal_force_error) > 1e-5:
        raise ValueError(f'Recomputed reactions disagree with recorded right-limit forces: {maximum_right_force_error:g}')

    force_integrands = -np.einsum('naik,nak->nai', yf[:last], filtered_d)
    corrected_filtered_residual = total_filtered_residual[:last]+filtered_d
    diagnostics = []
    for arm, end in enumerate(ends):
        rebuilt_g, intervals = _integral(gram_integrands[:, arm], times, end, window)
        rebuilt_p, _ = _integral(p_integrands[:, arm], times, end, window)
        total, _ = _integral(total_integrands[:, arm], times, end, window)
        force_part, _ = _integral(force_integrands[:, arm], times[:last], end, window)
        remaining = total-force_part
        if not len(intervals):
            raise ValueError('No original historical interval is available for diagnosis.')
        start = int(intervals[0]-1)
        indices = np.arange(start, end+1)
        force_error = arrays['internal_force'][indices, arm]-desired[indices, arm]
        norm_total, norm_remaining = float(np.linalg.norm(total)), float(np.linalg.norm(remaining))
        pieces = []
        # Split the actual accepted integration intervals into 2 s time bins;
        # these are descriptive contributions, never an online window rule.
        bin_ids = np.floor((times[intervals-1]-times[start]+1e-12)/2.).astype(int)
        for number in np.unique(bin_ids):
            chosen = intervals[bin_ids == number]
            weights = .5*(times[chosen]-times[chosen-1])
            contribution = np.sum(weights[:, None]*(total_integrands[chosen, arm]
                                                     +total_integrands[chosen-1, arm]), axis=0)
            pieces.append(dict(start_s=float(times[chosen[0]-1]), end_s=float(times[chosen[-1]]),
                               total_residual_vector=contribution.tolist(),
                               total_residual_norm=float(np.linalg.norm(contribution))))
        diagnostics.append(dict(
            arm=('left', 'right')[arm], frozen=reported_ie[arm] is not None,
            first_ie_s=first_ie[arm], evaluated_at_s=float(times[end]),
            nominal_window_start_s=max(0., float(times[end]-window)),
            sampled_window_start_s=float(times[start]), interval_count=len(intervals),
            rebuilt_gram_max_error=float(np.max(np.abs(rebuilt_g-gram[arm]))),
            rebuilt_p_max_error=float(np.max(np.abs(rebuilt_p-p[arm]))),
            rebuilt_total_vs_saved_residual_max_error=float(np.max(np.abs(total-stored_residual[arm]))),
            total_residual_vector=total.tolist(), total_residual_norm=norm_total,
            internal_force_residual_vector=force_part.tolist(),
            internal_force_residual_norm=float(np.linalg.norm(force_part)),
            remaining_residual_vector=remaining.tolist(), remaining_residual_norm=norm_remaining,
            remaining_norm_divided_by_total_norm=norm_remaining/norm_total if norm_total > 0 else None,
            filtered_transformed_residual_rms_before=np.sqrt(np.mean(
                total_filtered_residual[indices, arm]**2, axis=0)).tolist(),
            filtered_transformed_residual_rms_after_force_decomposition=np.sqrt(np.mean(
                corrected_filtered_residual[indices, arm]**2, axis=0)).tolist(),
            actual_internal_force_error_bias=force_error.mean(axis=0).tolist(),
            actual_internal_force_error_rms=np.sqrt(np.mean(force_error**2, axis=0)).tolist(),
            interval_residual_contributions=pieces))

    result = dict(
        case=summary['case'], scope='Offline decomposition of the original local JAS regression residual',
        compensated_learning=False, first_ie_recomputed_from_original_raw_gram=first_ie,
        filter_rate_s_inv=rate, original_window_s=window, original_raw_ie_threshold=threshold,
        maximum_replayed_vs_recorded_right_force_error=maximum_right_force_error,
        maximum_replayed_vs_recorded_right_internal_force_error=maximum_right_internal_force_error,
        maximum_actual_endpoint_force_jump=maximum_left_right_jump,
        method='d=a^(-1)(fI-fd); exact FOH filter from t=0 with previous right/current held-input left limits; zeta_force=-integral(Yf*d_f) on each original accepted window.',
        reaction_oracle='mj_forward at recorded qpos/qvel, with recorded current/previous applied torque. No mj_step, trajectory reset, learner input modification, or qacc reading.',
        remaining_interpretation='Remainder combines finite weld/local Euler chart differences, sampled filtering and quadrature, numerical model derivatives, and reaction-solver tolerances; these sources are not separately identified.',
        limits='Descriptive sampled decomposition. A small remainder does not establish zero total residual or exact truth convergence; G^-1*zeta is not the general original full-loop parameter limit.',
        channel_units='Filtered residuals use transformed generalized coordinates and mixed units. Actual internal-force errors are force N and generalized Euler moment Nm, as in the original recorded diagnostics.',
        verified_model_and_kinematic_sources=verified_sources,
        diagnostic_source_sha256=_sha256(Path(__file__)),
        input_sha256={name: _sha256(directory/name) for name in
                      ('trajectory.npz', 'learning_state.npz', 'summary.json', 'source_manifest.json')},
        arms=diagnostics)
    (directory/'residual_diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE/'results_original_retrial')
    parser.add_argument('--case', nargs='*', default=['all'])
    args = parser.parse_args()
    root = args.results.resolve()
    if (root/'trajectory.npz').exists():
        directories = [root]
    elif not args.case or args.case == ['all']:
        directories = sorted(p.parent for p in root.glob('*/source_manifest.json'))
    else:
        directories = [root/name for name in args.case]
    if not directories:
        raise SystemExit(f'No complete original-method cases found below {root}')
    for directory in directories:
        result = diagnose_case(directory)
        totals = [arm['total_residual_norm'] for arm in result['arms']]
        remaining = [arm['remaining_residual_norm'] for arm in result['arms']]
        print(f'{directory.name}: total zeta {totals}; remaining after offline force decomposition {remaining}', flush=True)


if __name__ == '__main__':
    main()
