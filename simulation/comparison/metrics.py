"""Raw comparison metrics; no controller, learner, or physics dependencies.

All saved samples belong to the reported interval. Nonfinite observations are
disclosed and invalidate the affected metric rather than being filtered out.
"""
import numpy as np


def sustained_entry_time(times, inside, hold_s=2., requested_end_s=None, completed=True):
    """First saved time in a final all-true suffix of at least ``hold_s``.

    The suffix must reach the requested experiment end. A terminated run is
    never reported as settled on the basis of its truncated final samples.
    No interpolation or transient successful island is treated as entry.
    """
    times = np.asarray(times, dtype=float)
    inside = np.asarray(inside, dtype=bool)
    if inside.ndim > 1:
        inside = np.all(inside, axis=tuple(range(1, inside.ndim)))
    if times.ndim != 1 or inside.shape != times.shape or hold_s < 0:
        raise ValueError('Entry requires matching one-dimensional times and conditions, hold_s>=0.')
    if not len(times) or not completed or not np.all(np.isfinite(times)):
        return None
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError('Entry times must strictly increase.')
    requested_end_s = float(times[-1]) if requested_end_s is None else float(requested_end_s)
    if times[-1] < requested_end_s-1e-9 or not inside[-1]:
        return None
    last_bad = np.flatnonzero(~inside)
    start = int(last_bad[-1]+1) if len(last_bad) else 0
    return float(times[start]) if times[-1]-times[start] >= hold_s-1e-12 else None


def tracking_entry_metrics(times, error, position_threshold_m=.00025,
                           orientation_threshold_rad=np.deg2rad(.05), hold_s=2.,
                           requested_end_s=None, completed=True):
    times, error = np.asarray(times), np.asarray(error)
    if error.shape != (len(times), 6):
        raise ValueError('Tracking errors must have shape [ticks,6].')
    finite = np.all(np.isfinite(error), axis=1)
    position = np.linalg.norm(error[:, :3], axis=1)
    orientation = np.linalg.norm(error[:, 3:], axis=1)
    kwargs = dict(hold_s=hold_s, requested_end_s=requested_end_s, completed=completed)
    position_inside = finite & (position <= position_threshold_m)
    orientation_inside = finite & (orientation <= orientation_threshold_rad)
    thresholds = np.r_[np.full(3, position_threshold_m), np.full(3, orientation_threshold_rad)]
    return dict(position_entry_s=sustained_entry_time(times, position_inside, **kwargs),
        orientation_coordinate_entry_s=sustained_entry_time(times, orientation_inside, **kwargs),
        simultaneous_entry_s=sustained_entry_time(times, position_inside & orientation_inside, **kwargs),
        componentwise_entry_s=[sustained_entry_time(times, finite & (np.abs(error[:, j]) <= thresholds[j]),
                                                    **kwargs) for j in range(6)],
        thresholds=dict(position_norm_m=float(position_threshold_m),
                        orientation_coordinate_norm_rad=float(orientation_threshold_rad)),
        hold_s=float(hold_s), completed=bool(completed),
        interpretation='First saved time after which all saved samples satisfy the criterion until the requested end, for at least hold_s; orientation uses Euler-coordinate error norm.')


def vector_statistics(actual, desired):
    actual, desired = np.asarray(actual), np.asarray(desired)
    error = actual-desired
    if actual.ndim != 2 or actual.shape != desired.shape or not len(actual):
        raise ValueError('Vector statistics require matching nonempty [ticks,components].')
    actual_mean, desired_mean, bias = actual.mean(axis=0), desired.mean(axis=0), error.mean(axis=0)
    ac, error_ac = actual-actual_mean, error-bias
    return dict(samples=len(actual), all_finite=bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(desired))),
        nonfinite_actual_samples=int(np.count_nonzero(~np.all(np.isfinite(actual), axis=1))),
        nonfinite_desired_samples=int(np.count_nonzero(~np.all(np.isfinite(desired), axis=1))),
        actual_mean=actual_mean, desired_mean=desired_mean, error_bias=bias,
        error_rms_components=np.sqrt(np.mean(error**2, axis=0)),
        error_vector_rms=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        actual_ac_rms_components=np.sqrt(np.mean(ac**2, axis=0)),
        actual_ac_vector_rms=float(np.sqrt(np.mean(np.sum(ac**2, axis=1)))),
        error_ac_rms_components=np.sqrt(np.mean(error_ac**2, axis=0)),
        error_ac_vector_rms=float(np.sqrt(np.mean(np.sum(error_ac**2, axis=1)))),
        actual_peak_to_peak_components=np.ptp(actual, axis=0),
        error_peak_to_peak_components=np.ptp(error, axis=0),
        sampled_max_abs_actual_components=np.max(np.abs(actual), axis=0),
        sampled_max_abs_error_components=np.max(np.abs(error), axis=0),
        sampled_max_actual_vector_norm=float(np.max(np.linalg.norm(actual, axis=1))),
        sampled_max_error_vector_norm=float(np.max(np.linalg.norm(error, axis=1))))


def force_statistics(times, actual, desired, interval=None):
    times, actual, desired = np.asarray(times), np.asarray(actual), np.asarray(desired)
    if actual.shape != (len(times), 2, 6) or actual.shape != desired.shape:
        raise ValueError('Force signals must have shape [ticks,2,6].')
    interval = (float(times[0]), float(times[-1])) if interval is None and len(times) else interval
    if interval is None:
        return dict(available=False, samples=0, reason='No recorded samples.')
    lo, hi = interval
    selected = (times >= lo-1e-12) & (times <= hi+1e-12)
    if not np.any(selected):
        return dict(available=False, samples=0, requested_interval_s=list(interval))
    return dict(available=True, interval_s=[float(times[selected][0]), float(times[selected][-1])],
        samples=int(np.count_nonzero(selected)),
        arms={name:dict(force_xyz_N=vector_statistics(actual[selected, arm, :3], desired[selected, arm, :3]),
                        euler_moment_xyz_Nm=vector_statistics(actual[selected, arm, 3:], desired[selected, arm, 3:]))
              for arm, name in enumerate(('left', 'right'))},
        interpretation='Uniform saved-tick statistics, inclusive endpoints; AC removes only within-window mean. Cartesian force N and generalized Euler-coordinate moment Nm are reported separately; peaks are saved-tick peaks.')


def torque_statistics(times, command, applied, limits):
    times, command, applied, limits = map(np.asarray, (times, command, applied, limits))
    if command.shape != applied.shape or command.shape != (len(times), len(limits)):
        raise ValueError('Torque signals require [ticks,motors] and one positive limit per motor.')
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError('Torque limits must be finite positive.')
    saturated = np.abs(command-applied) > 1e-12
    return dict(samples=len(times), all_finite=bool(np.all(np.isfinite(command)) and np.all(np.isfinite(applied))),
        raw_command_rms_Nm=np.sqrt(np.mean(command**2, axis=0)),
        applied_rms_Nm=np.sqrt(np.mean(applied**2, axis=0)),
        raw_command_peak_abs_Nm=np.max(np.abs(command), axis=0),
        applied_peak_abs_Nm=np.max(np.abs(applied), axis=0),
        maximum_raw_limit_ratio=float(np.max(np.abs(command)/limits)),
        maximum_applied_limit_ratio=float(np.max(np.abs(applied)/limits)),
        saturated_tick_count=int(np.count_nonzero(np.any(saturated, axis=1))),
        saturated_motor_tick_counts=np.count_nonzero(saturated, axis=0),
        saturation_fraction=float(np.mean(np.any(saturated, axis=1))),
        raw_command_total_variation_Nm=np.sum(np.abs(np.diff(command, axis=0)), axis=0),
        applied_total_variation_Nm=np.sum(np.abs(np.diff(applied, axis=0)), axis=0),
        expected_clipping_max_error_Nm=float(np.max(np.abs(applied-np.clip(command, -limits, limits)))),
        interpretation='Raw command and physical clipped command both retained. Saturation is a performance outcome; incorrect clipping or omitted samples is a validation error.')
