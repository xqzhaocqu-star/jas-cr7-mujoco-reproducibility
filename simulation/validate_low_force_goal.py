#!/usr/bin/env python3
"""Independently assess force performance and bounded parameters from raw logs.

This complements, and never invokes or changes, validate_original_retrial.py.
Only low_force_assessment.json and performance_assessment.json are written.
Force ratios are descriptive: neither improvement in every component nor zero
parameter drift is required. Box invariance is not an exponential-envelope test.
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

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CASES = ('nominal', 'nonmonotonic', 'large_initial_error')
RAW_FIELDS = ('t', 'internal_force', 'desired_internal_force', 'theta',
              'theta_control', 'prediction_error', 'regressor', 'surface',
              'reference')
RATIO_FIELDS = ('error_rms_components', 'error_vector_rms',
                'actual_ac_rms_components', 'actual_ac_vector_rms',
                'error_ac_rms_components', 'error_ac_vector_rms',
                'actual_peak_to_peak_components', 'error_peak_to_peak_components',
                'sampled_max_abs_actual_components', 'sampled_max_abs_error_components',
                'sampled_max_actual_vector_norm', 'sampled_max_error_vector_norm',
                'absolute_error_bias_components')


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Assessment:
    def __init__(self, case):
        self.report = dict(case=case, passed=False, checks={}, failures=[],
            scope='Independent raw-log performance/configuration/boundedness assessment; '
                  'complements the original 54 physical and learning checks. '
                  'No physics is executed. No measured-force feedback is introduced. '
                  'No empirical exponential envelope is certified.')

    def check(self, name, passed, **details):
        self.report['checks'][name] = dict(passed=bool(passed), **plain(details))
        if not passed:
            self.report['failures'].append(name)

    def finish(self, directory):
        self.report['passed'] = not self.report['failures']
        (directory/'low_force_assessment.json').write_text(
            json.dumps(plain(self.report), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        return self.report


def load_case(directory):
    summary = json.loads((directory/'summary.json').read_text())
    with np.load(directory/'trajectory.npz', allow_pickle=False) as source:
        missing = sorted(set(RAW_FIELDS)-set(source.files))
        if missing:
            raise ValueError('Missing raw channels: '+', '.join(missing))
        arrays = {key: source[key] for key in RAW_FIELDS}
    return summary, arrays


def positive(value, size=None, scalar=False):
    array = np.asarray(value, dtype=float)
    shape_ok = array.ndim == 0 if scalar else (size is None or array.shape == (size,))
    return bool(shape_ok and np.all(np.isfinite(array)) and np.all(array > 0))


def scalar_value(settings, name):
    array = np.asarray(settings[name], dtype=float)
    return float(array) if array.ndim == 0 else float('nan')


def check_configuration(audit, summary, manifest):
    cfg = summary['settings']
    positive_gains = all(positive(cfg[name], 6) or positive(cfg[name], scalar=True)
                         for name in ('ka', 'k2', 'k3'))
    scalar_surface = all(positive(cfg[name], scalar=True) for name in ('lam1', 'lam2', 'delta'))
    delta = scalar_value(cfg, 'delta')
    audit.check('positive_original_feedback_and_scalar_surface_gains',
        positive_gains and scalar_surface and 0 < delta < 1,
        Ka=cfg['ka'], K2=cfg['k2'], K3=cfg['k3'],
        Lambda1=cfg['lam1'], Lambda2=cfg['lam2'], delta=cfg['delta'],
        supported_surface='Positive scalar Lambda1/Lambda2 and 0<delta<1; original equations retained.')
    alpha, beta = scalar_value(cfg, 'alpha'), scalar_value(cfg, 'beta')
    v1, v2 = scalar_value(cfg, 'upsilon1'), scalar_value(cfg, 'upsilon2')
    paired = (0 < alpha < 1 and beta > 1 and abs(alpha+beta-2) < 1e-12
              and 1 < v1 < 2 and 0 < v2 < 1 and abs(v1+v2-2) < 1e-12)
    audit.check('paper_paired_terminal_and_reaching_exponents', paired,
        alpha=alpha, beta=beta, upsilon1=v1, upsilon2=v2,
        phi=1/(1-alpha) if 0 < alpha < 1 else None,
        psi=2/(v1-1) if v1 > 1 else None)
    prior_low, prior_high = scalar_value(cfg, 'prior_lower'), scalar_value(cfg, 'prior_upper')
    positive_learning = (positive(cfg['adaptation_gain'], 8)
                         and positive(cfg['gamma'], scalar=True)
                         and all(positive(cfg[name], scalar=True)
                                 for name in ('filter_rate', 'window', 'omega', 'rho_hat')))
    audit.check('positive_constant_learning_filter_ie_and_rho_configuration',
        positive_learning and np.isfinite(prior_low+prior_high) and prior_low < prior_high,
        Upsilon_diagonal=cfg['adaptation_gain'], gamma=cfg['gamma'],
        filter_rate=cfg['filter_rate'], window=cfg['window'], omega=cfg['omega'],
        rho_hat=cfg['rho_hat'], prior=[prior_low, prior_high])
    audit.check('original_sign_feedback_declared', cfg.get('robust_mode') == 'paper_sign',
        robust_mode=cfg.get('robust_mode'),
        robust_diagonal_implied_by_prior=[prior_high-prior_low]*8,
        interpretation='R=diag(prior_upper-prior_lower); no reduced uncertainty width or smoothed sign is accepted.')

    # Structural checks cover the actually imported original controller paths.
    # The original validator separately reconstructs the complete torque/law.
    paths = ('simulation/jas/controller.py', 'simulation/jas/learning.py',
             'simulation/original_retrial/controller.py',
             'simulation/original_retrial/learning.py')
    bad_hashes, mutations, trees = [], [], {}
    gain_names = {'gain', 'gamma', 'adaptation_gain', 'ka', 'k2', 'k3',
                  'lam1', 'lam2', 'delta', 'radius', 'rho_hat'}
    for relative in paths:
        path = PROJECT/relative
        if not path.is_file() or relative not in manifest or sha256(path) != manifest[relative]:
            bad_hashes.append(relative)
            continue
        tree = ast.parse(path.read_text())
        trees[relative] = tree
        for function in [item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef)
                         and item.name != '__init__']:
            for node in ast.walk(function):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if any(isinstance(part, ast.Attribute) and part.attr in gain_names
                               for part in ast.walk(target)):
                            mutations.append(dict(file=relative, line=node.lineno))
    def expression_present(tree, target_name, expected):
        expected_dump = ast.dump(ast.parse(expected, mode='eval').body)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [item.attr if isinstance(item, ast.Attribute)
                       else item.id if isinstance(item, ast.Name) else '' for item in node.targets]
            if target_name in targets and ast.dump(node.value) == expected_dump:
                return True
        return False
    radius_ok = expression_present(trees.get(paths[0], ast.Module(body=[], type_ignores=[])),
        'radius', 'np.full(8, (c.prior_upper-c.prior_lower)/2)')
    robust_ok = expression_present(trees.get(paths[2], ast.Module(body=[], type_ignores=[])),
        'robust', 'regressor.T@(2*self.radius*np.sign(w))')
    audit.check('source_bound_original_R_and_time_constant_gains',
        not bad_hashes and not mutations and radius_ok and robust_ok,
        source_sha_mismatches=bad_hashes, outside_initialization_gain_mutations=mutations,
        prior_radius_expression_verified=radius_ok, original_sign_R_expression_verified=robust_ok,
        scope='Structural source/configuration check; full original law and sensing remain the separate original validation responsibility.')


def window_definitions(times, summary):
    start, end = float(times[0]), float(times[-1])
    excitation_end = float(summary['settings'].get('excitation_duration', end))
    return {'full': (start, end),
            'motion_after_2s': (max(start, 2.), min(end, excitation_end)),
            'final_two_seconds': (max(start, end-2.), end)}


def force_group(actual, desired):
    error = actual-desired
    mean, bias = actual.mean(axis=0), error.mean(axis=0)
    ac, error_ac = actual-mean, error-bias
    return dict(actual_mean=mean, desired_mean=desired.mean(axis=0),
        error_bias=bias, absolute_error_bias_components=np.abs(bias),
        error_rms_components=np.sqrt(np.mean(error**2, axis=0)),
        error_vector_rms=np.sqrt(np.mean(np.sum(error**2, axis=1))),
        actual_ac_rms_components=np.sqrt(np.mean(ac**2, axis=0)),
        actual_ac_vector_rms=np.sqrt(np.mean(np.sum(ac**2, axis=1))),
        error_ac_rms_components=np.sqrt(np.mean(error_ac**2, axis=0)),
        error_ac_vector_rms=np.sqrt(np.mean(np.sum(error_ac**2, axis=1))),
        actual_peak_to_peak_components=np.ptp(actual, axis=0),
        error_peak_to_peak_components=np.ptp(error, axis=0),
        sampled_max_abs_actual_components=np.max(np.abs(actual), axis=0),
        sampled_max_abs_error_components=np.max(np.abs(error), axis=0),
        sampled_max_actual_vector_norm=np.max(np.linalg.norm(actual, axis=1)),
        sampled_max_error_vector_norm=np.max(np.linalg.norm(error, axis=1)))


def force_statistics(arrays, interval):
    lo, hi = interval
    selected = (arrays['t'] >= lo-1e-12) & (arrays['t'] <= hi+1e-12)
    if hi <= lo or np.count_nonzero(selected) < 2:
        return dict(available=False, requested_interval_s=interval,
                    reason='Fewer than two saved samples or no positive-duration interval.')
    actual = arrays['internal_force'][selected]
    desired = arrays['desired_internal_force'][selected]
    return dict(available=True, interval_s=[float(arrays['t'][selected][0]),
                                         float(arrays['t'][selected][-1])],
        samples=int(np.count_nonzero(selected)), arms={
            name: dict(force_xyz_N=force_group(actual[:, arm, :3], desired[:, arm, :3]),
                       euler_moment_xyz_Nm=force_group(actual[:, arm, 3:], desired[:, arm, 3:]))
            for arm, name in enumerate(('left', 'right'))})


def ratio(candidate, baseline):
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    result = np.empty(candidate.shape, dtype=object)
    for index in np.ndindex(candidate.shape):
        result[index] = (float(candidate[index]/baseline[index])
                         if abs(baseline[index]) > 1e-15 else None)
    return result.item() if result.ndim == 0 else result.tolist()


def compare_force_windows(candidate_arrays, reference_arrays, windows):
    result = {}
    for name, interval in windows.items():
        candidate = force_statistics(candidate_arrays, interval)
        baseline = force_statistics(reference_arrays, interval)
        ratios = {}
        if candidate['available'] and baseline['available']:
            for arm in ('left', 'right'):
                ratios[arm] = {}
                for unit in ('force_xyz_N', 'euler_moment_xyz_Nm'):
                    ratios[arm][unit] = {field: ratio(candidate['arms'][arm][unit][field],
                                                     baseline['arms'][arm][unit][field])
                                        for field in RATIO_FIELDS}
        result[name] = dict(interval_s=interval, candidate=candidate,
                            baseline_matched=baseline, ratio_candidate_over_baseline=ratios)
    return result


def interpolate_values(times, values, target_times):
    flat = values.reshape(len(times), -1)
    interpolated = np.column_stack([np.interp(target_times, times, flat[:, index])
                                    for index in range(flat.shape[1])])
    return interpolated.reshape((len(target_times),)+values.shape[1:])


def check_matched_task(audit, arrays, baseline, summary, baseline_summary):
    times = arrays['t']
    covered = bool(times[0] >= baseline['t'][0]-1e-12 and times[-1] <= baseline['t'][-1]+1e-12)
    desired = interpolate_values(baseline['t'], baseline['desired_internal_force'], times)
    reference = interpolate_values(baseline['t'], baseline['reference'], times)
    desired_error = float(np.max(np.abs(desired-arrays['desired_internal_force'])))
    reference_error = float(np.max(np.abs(reference-arrays['reference'])))
    # Interpolation on a different control grid has the ordinary O(h^2) error;
    # verify the declared reference profile and duration in addition to values.
    declared_same = all(summary['settings'].get(key) == baseline_summary['settings'].get(key)
                        for key in ('excitation_profile', 'excitation_duration', 'internal_force'))
    scenario_same = (summary.get('case') == baseline_summary.get('case')
                     and summary.get('scenario') == baseline_summary.get('scenario'))
    prior_same = all(summary['settings'].get(key) == baseline_summary['settings'].get(key)
                     for key in ('prior_lower', 'prior_upper'))
    audit.check('baseline_covers_same_candidate_time_and_task_targets',
        covered and declared_same and scenario_same and desired_error <= 1e-12 and reference_error <= 2e-7,
        candidate_interval_s=[float(times[0]), float(times[-1])],
        baseline_available_interval_s=[float(baseline['t'][0]), float(baseline['t'][-1])],
        maximum_desired_force_or_moment_discrepancy=desired_error,
        maximum_reference_coordinate_interpolation_difference=reference_error,
        same_reference_and_force_declaration=declared_same,
        same_initial_condition_and_scenario_declaration=scenario_same,
        reference_interpolation_tolerance=2e-7,
        interpretation='Baseline force statistics are cut to the candidate windows, not to the baseline final two seconds.')
    audit.check('same_prior_uncertainty_task_for_robust_comparison', prior_same,
        candidate_prior=[summary['settings'].get(key) for key in ('prior_lower', 'prior_upper')],
        baseline_prior=[baseline_summary['settings'].get(key) for key in ('prior_lower', 'prior_upper')])


def theta_statistics(audit, arrays, summary):
    times, theta, old = arrays['t'], arrays['theta'], arrays['theta_control']
    cfg = summary['settings']
    lower, upper = float(cfg['prior_lower']), float(cfg['prior_upper'])
    truth = np.asarray(summary['true_parameters_offline_only'], dtype=float)
    finite = bool(np.all(np.isfinite(theta)) and np.all(np.isfinite(old)))
    inside = bool(np.all(theta >= lower-1e-12) and np.all(theta <= upper+1e-12)
                  and np.all(old >= lower-1e-12) and np.all(old <= upper+1e-12))
    truth_inside = bool(np.all(truth >= lower) and np.all(truth <= upper))
    audit.check('raw_theta_finite_and_within_declared_covering_prior',
        finite and inside and truth_inside, estimates_finite=finite,
        estimates_within_prior=inside, offline_truth_within_prior=truth_inside,
        prior=[lower, upper],
        interpretation='Finite-run boundedness and invariance; not a test or proof of an exponential error envelope.')
    tracking_drive = np.einsum('napc,nac->nap', arrays['regressor'], arrays['surface'])
    direction = -tracking_drive[1:]+float(cfg['gamma'])*arrays['prediction_error'][1:]
    active = (((old[1:] >= upper) & (direction > 0))
              | ((old[1:] <= lower) & (direction < 0)))
    final = times >= max(float(times[0]), float(times[-1])-2.)-1e-12
    centered_time = times[final]-times[final].mean()
    method = summary.get('learning_integrator', cfg.get('learning_integrator', 'explicit_euler'))
    explicit = method in ('explicit_euler', 'projected_euler')
    clip_data = None
    if explicit:
        dt = np.diff(times)[:, None, None]
        trial = old[1:]+dt*np.asarray(cfg['adaptation_gain'])*np.where(active, 0., direction)
        clipped = (trial < lower) | (trial > upper)
        overshoot = np.maximum(lower-trial, 0.)+np.maximum(trial-upper, 0.)
        predicted = np.clip(trial, lower, upper)
        audit.check('finite_raw_learning_direction_and_preclip_euler_trial',
            np.all(np.isfinite(direction)) and np.all(np.isfinite(trial)),
            scope='Rejects arithmetic overflow before projection; legitimate finite box overshoot is reported separately.')
        step_error = float(np.max(np.abs(predicted-theta[1:])))
        clip_counts = np.count_nonzero(np.any(clipped, axis=2), axis=0)
        recorded = summary.get('numerical_projection_counts')
        audit.check('raw_explicit_projection_and_numerical_clip_counts_reconstructed',
            step_error <= 3e-10 and (recorded is None or np.array_equal(clip_counts, recorded)),
            maximum_parameter_step_discrepancy=step_error, rebuilt_clip_tick_counts=clip_counts,
            recorded_clip_tick_counts=recorded,
            scope='YS reconstructed from raw Y/S; logged pre-update epsilon is independently audited by the original validator.')
        clip_data = (clipped, overshoot, clip_counts)
    arms = {}
    for arm, name in enumerate(('left', 'right')):
        error = theta[:, arm]-truth[arm]
        final_theta = theta[final, arm]
        drift = final_theta[-1]-final_theta[0]
        slope = (np.sum(centered_time[:, None]*final_theta, axis=0)/np.sum(centered_time**2)
                 if np.sum(centered_time**2) > 0 else np.full(theta.shape[-1], np.nan))
        item = dict(minimum_estimate_componentwise=theta[:, arm].min(axis=0),
            maximum_estimate_componentwise=theta[:, arm].max(axis=0),
            global_minimum=float(theta[:, arm].min()), global_maximum=float(theta[:, arm].max()),
            offline_maximum_absolute_true_error_componentwise=np.max(np.abs(error), axis=0),
            offline_maximum_true_error_L2=float(np.max(np.linalg.norm(error, axis=1))),
            offline_final_true_error_L2=float(np.linalg.norm(error[-1])),
            final_two_second_endpoint_drift=drift,
            final_two_second_endpoint_drift_L2=float(np.linalg.norm(drift)),
            final_two_second_linear_slope_per_s=slope,
            final_two_second_peak_to_peak_components=np.ptp(final_theta, axis=0),
            maximum_parameter_step_components=np.max(np.abs(np.diff(theta[:, arm], axis=0)), axis=0),
            final_two_second_maximum_parameter_step_components=np.max(np.abs(np.diff(final_theta, axis=0)), axis=0),
            lower_face_saved_tick_counts=np.count_nonzero(theta[:, arm] <= lower+1e-12, axis=0),
            upper_face_saved_tick_counts=np.count_nonzero(theta[:, arm] >= upper-1e-12, axis=0),
            outward_tangent_projection_tick_count=int(np.count_nonzero(np.any(active[:, arm], axis=1))),
            outward_tangent_projection_component_counts=np.count_nonzero(active[:, arm], axis=0),
            outward_tangent_projection_tick_fraction=float(np.mean(np.any(active[:, arm], axis=1))))
        if clip_data is not None:
            clipped, overshoot, counts = clip_data
            item.update(numerical_euler_clip_tick_count=int(counts[arm]),
                numerical_euler_clip_component_counts=np.count_nonzero(clipped[:, arm], axis=0),
                maximum_preclip_box_overshoot_components=overshoot[:, arm].max(axis=0),
                maximum_preclip_box_overshoot=float(overshoot[:, arm].max()))
        arms[name] = item
    return dict(prior=[lower, upper], learning_integrator=method,
        final_window_s=[float(times[final][0]), float(times[final][-1])], arms=arms,
        clip_metric_scope='Actual explicit Euler trial overshoot.' if explicit else
            'No explicit trial overshoot metric for an implicit integrator; original validator must check box VI.',
        interpretation='Projection activity, nonzero true error and nonzero finite-time drift are reported; '
                       'none is automatically classified as parameter divergence.')


def compare_theta_to_baseline(arrays, baseline, baseline_summary, theta):
    """Use the same elapsed interval for parameter metrics in a short trial."""
    keep = ((baseline['t'] >= arrays['t'][0]-1e-12)
            & (baseline['t'] <= arrays['t'][-1]+1e-12))
    clipped_baseline = {key: value[keep] for key, value in baseline.items()}
    summary = dict(baseline_summary)
    # Full-run counters cannot be compared with a shortened baseline interval.
    summary.pop('numerical_projection_counts', None)
    auxiliary = Assessment('baseline_matched_interval')
    baseline_theta = theta_statistics(auxiliary, clipped_baseline, summary)
    fields = ('offline_maximum_true_error_L2', 'offline_final_true_error_L2',
              'final_two_second_endpoint_drift_L2', 'maximum_parameter_step_components',
              'final_two_second_maximum_parameter_step_components')
    ratios = {arm: {field: ratio(theta['arms'][arm][field], baseline_theta['arms'][arm][field])
                    for field in fields} for arm in ('left', 'right')}
    return dict(baseline_matched=baseline_theta,
        ratio_candidate_over_baseline=ratios,
        baseline_raw_metric_checks_passed=not auxiliary.report['failures'],
        baseline_raw_metric_checks=auxiliary.report['checks'],
        interpretation='Matched elapsed-time comparison. Lower true error or zero final drift is not required for PASS.')


def assess_case(directory, baseline_directory, case, main_directory=None):
    audit = Assessment(case)
    summary, arrays = load_case(directory)
    baseline_summary, baseline = load_case(baseline_directory)
    manifest = json.loads((directory/'source_manifest.json').read_text())
    n = len(arrays['t'])
    expected = {'internal_force': (n, 2, 6), 'desired_internal_force': (n, 2, 6),
                'theta': (n, 2, 8), 'theta_control': (n, 2, 8),
                'prediction_error': (n, 2, 8), 'regressor': (n, 2, 8, 6),
                'surface': (n, 2, 6), 'reference': (n, 6)}
    bad_shapes = {key: dict(actual=arrays[key].shape, expected=shape)
                  for key, shape in expected.items() if arrays[key].shape != shape}
    bad_finite = [key for key in RAW_FIELDS if not np.all(np.isfinite(arrays[key]))]
    grid_ok = n > 1 and np.all(np.diff(arrays['t']) > 0)
    audit.check('finite_complete_raw_performance_channels', not bad_shapes and not bad_finite and grid_ok,
        bad_shapes=bad_shapes, nonfinite_channels=bad_finite, monotonic_time=bool(grid_ok))
    if bad_shapes or bad_finite or not grid_ok:
        return audit.finish(directory)
    check_configuration(audit, summary, manifest)
    check_matched_task(audit, arrays, baseline, summary, baseline_summary)
    windows = window_definitions(arrays['t'], summary)
    theta = theta_statistics(audit, arrays, summary)
    theta_comparison = compare_theta_to_baseline(arrays, baseline, baseline_summary, theta)
    audit.check('matched_baseline_parameter_metrics_consistent',
        theta_comparison['baseline_raw_metric_checks_passed'],
        scope='Independent parameter metric checks on the same elapsed baseline interval; full-run clip counters are not reused for a short interval.')
    audit.report.update(source_directory=str(directory.resolve()),
        baseline_directory=str(baseline_directory.resolve()),
        assessment_source_sha256=sha256(Path(__file__)),
        input_sha256={name: sha256(directory/name)
                      for name in ('trajectory.npz', 'summary.json', 'source_manifest.json')},
        baseline_input_sha256={name: sha256(baseline_directory/name)
                               for name in ('trajectory.npz', 'summary.json', 'source_manifest.json')},
        control_dt_s=summary['control_dt_s'], physics_dt_s=summary['physics_dt_s'],
        windows=compare_force_windows(arrays, baseline, windows),
        theta=theta,
        theta_comparison=theta_comparison,
        methodology=dict(weighting='Uniform saved-control-tick means; inclusive interval endpoints; no filtering or smoothing.',
            units='force_xyz_N: Cartesian N; euler_moment_xyz_Nm: generalized Nm conjugate to XYZ Euler angles.',
            AC='Signal minus its within-window mean; includes all non-DC variation, not solely high-frequency chatter.',
            peaks='Saved control-tick extrema; no claim about unrecorded physics-substep force peaks.',
            ratios='Candidate divided by matched-time baseline; null when denominator is zero. Descriptive, without requiring every component to improve.',
            windows='full=start-to-end; motion_after_2s=2-to-min(excitation_end,run_end); final_two_seconds=run_end-2-to-run_end. Baseline uses these exact time intervals.',
            residual_bound='No zero-truth-error or zero-drift requirement. Prior invariance is not empirical verification of the residual-exponential envelope.'))
    original_report = directory/'validation.json'
    audit.report['separate_original_validation'] = (
        dict(available=True, passed=json.loads(original_report.read_text()).get('passed'),
             path=str(original_report.resolve())) if original_report.is_file()
        else dict(available=False, interpretation='This assessment does not replace the separate original physical/learning validation.'))
    if main_directory is not None:
        main_summary, main_arrays = load_case(main_directory)
        audit.report['fine_sensitivity_against_main'] = dict(
            reference_directory=str(main_directory.resolve()),
            reference_control_dt_s=main_summary['control_dt_s'],
            windows=compare_force_windows(arrays, main_arrays, windows),
            interpretation='Descriptive force-statistic sensitivity across the actual saved sample grids; '
                           'force waveform convergence is not inferred from pose similarity.')
    return audit.finish(directory)


def compact_case(report, path):
    final = report.get('windows', {}).get('final_two_seconds', {})
    arms = final.get('candidate', {}).get('arms', {})
    ratios = final.get('ratio_candidate_over_baseline', {})
    metrics = {}
    for arm in arms:
        metrics[arm] = {unit: {name: dict(value=arms[arm][unit][name],
                                        ratio_to_matched_baseline=ratios.get(arm, {}).get(unit, {}).get(name))
                              for name in ('error_vector_rms', 'actual_ac_vector_rms',
                                           'sampled_max_error_vector_norm')}
                        for unit in ('force_xyz_N', 'euler_moment_xyz_Nm')}
    return dict(case=report['case'], passed=report['passed'], failures=report['failures'],
                assessment_path=str(path.resolve()), final_two_second_metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--case', choices=CASES+('all',), default='all')
    args = parser.parse_args()
    names = CASES if args.case == 'all' else (args.case,)
    # A separate fine-root invocation avoids reprocessing an already assessed
    # main nominal case.  Its sibling main result supplies force sensitivity.
    fine_root = args.results.name == 'step_refinement'
    jobs = [(name, args.results/name, args.baseline/name,
             args.results.parent/name if fine_root else None) for name in names]
    # Process fine explicitly from its own root, after its original validation.
    # This prevents a finishing fine run from being picked up twice while a
    # main-case assessment and a fine completion watcher run independently.
    fine_status = 'standalone_fine_root' if fine_root else 'main_cases_only'
    reports = []
    for name, directory, baseline, main_directory in jobs:
        if not directory.is_dir():
            reports.append(dict(case=name, passed=False, failures=['missing_case_directory'],
                                assessment_path=str(directory/'low_force_assessment.json')))
            print(name+': FAIL missing_case_directory', flush=True)
            continue
        try:
            report = assess_case(directory, baseline, name, main_directory)
        except Exception as exc:
            audit = Assessment(name)
            audit.check('assessment_completed', False, exception_type=type(exc).__name__, message=str(exc))
            report = audit.finish(directory)
        reports.append(compact_case(report, directory/'low_force_assessment.json'))
        print(name+': '+('PASS' if report['passed'] else 'FAIL '+', '.join(report['failures'])), flush=True)
    aggregate = dict(passed=all(item['passed'] for item in reports), cases=reports,
        optional_fine_assessment=fine_status,
        acceptance_scope='PASS means raw/configuration/task-comparison/boundedness checks pass; '
                         'performance ratios remain descriptive. The separate original validator '
                         'must establish physics and full-law consistency. No empirical exponential envelope is certified.')
    (args.results/'performance_assessment.json').write_text(
        json.dumps(plain(aggregate), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return 0 if aggregate['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
