#!/usr/bin/env python3
"""Plot original JAS sign-control and full composite-learning retrials.

The first seven figure types reuse the original experiment plotter. Figure 08
shows recorded offline truth error, eight component errors, final-two-second
parameter drift and the actual raw-Gram IE test. No exponential-to-truth curve
or pure-history convergence model is imposed. All raw samples are retained.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from plot_results import (CASE_TITLES, COMPONENT_COLORS, HAND_COLORS, HERE,
                          _diagnostics, _errors, _internal_force, _learning,
                          _read_json, _save, _style, _time_axes, _torques,
                          _trajectory, _transformed)

ARRAY_KEYS = ('t', 'object_pose', 'reference', 'error', 'upper', 'lower',
              'surface', 'xi', 'torque_applied', 'torque_command', 'theta',
              'lambda_min', 'internal_force', 'desired_internal_force',
              'weld_residual', 'barrier_margin', 'sigma_min', 'contact_count')
PARAMETER_COLORS = (*COMPONENT_COLORS, '#00a5a5', '#a65628')


def _title(summary, directory):
    settings = summary['settings']
    case = CASE_TITLES.get(summary.get('case', directory.name), directory.name.replace('_', ' ').title())
    return (f'ORIGINAL JAS LAW RETRIAL | {case} | paper sign | '
            f'−YS + γε | constant diagonal Υ | γ={float(settings["gamma"]):g}')


def _truth(case_directory, summary):
    # Completed offline evaluation records are the only truth source here.
    if 'true_parameters_offline_only' in summary:
        truth = np.asarray(summary['true_parameters_offline_only'], dtype=float)
    else:
        with np.load(case_directory/'learning_state.npz', allow_pickle=False) as state:
            truth = state['true_parameters_offline_only']
    if truth.shape != (2, 8) or not np.isfinite(truth).all():
        raise ValueError('Offline evaluation must provide finite two-arm, eight-parameter truth')
    return truth


def _raw_ie_axis(ax, arrays, summary):
    t, values = arrays['t'], arrays['lambda_min']
    omega = float(summary['settings']['omega'])
    linthresh = max(.05*omega, 1e-14)
    descriptions = []
    for hand, name in enumerate(('Left', 'Right')):
        color = HAND_COLORS[hand]
        ax.plot(t, values[:, hand], color=color, label=f'{name} recorded raw λmin(G)')
        te = summary.get('ie_detection_times_s', [None, None])[hand]
        descriptions.append(f'{name}: '+(f'IE/freeze at {te:g} s' if te is not None else 'IE not detected'))
        if te is not None:
            ax.axvline(te, color=color, linestyle=':', linewidth=1.)
    ax.axhline(omega, color='#222222', linestyle='--', linewidth=1., label=f'Raw IE threshold ω={omega:g}')
    ax.set_yscale('symlog', linthresh=linthresh)
    low, high = float(np.min(values)), max(omega, float(np.max(values)))
    ax.set_ylim(min(0., low)-.05*linthresh, 1.5*high)
    powers = np.arange(int(np.ceil(np.log10(linthresh))), int(np.floor(np.log10(1.5*high)))+1)
    if len(powers) > 7:
        powers = powers[np.linspace(0, len(powers)-1, 7).round().astype(int)]
    positive = 10.**powers
    negative = -positive[positive <= -low][::-1]
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.r_[negative, 0., positive]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: '0' if v == 0 else rf"${'-' if v < 0 else ''}10^{{{int(np.round(np.log10(abs(v))))}}}$"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set(title='; '.join(descriptions), ylabel=r'$\lambda_{\min}(G)$', xlabel='Time (s)', xlim=(t[0], t[-1]))
    ax.legend(loc='best', ncol=3, fontsize=7.5)


def _parameter_error(arrays, out, title, summary, truth):
    t, theta = arrays['t'], arrays['theta']
    error = theta-truth[None, :, :]
    tail = t >= max(t[0], t[-1]-2.)
    first_tail = int(np.flatnonzero(tail)[0])
    fig = plt.figure(figsize=(11.5, 12.0))
    grid = fig.add_gridspec(4, 2, height_ratios=(1., 1.05, 1.05, .85))
    for hand, name in enumerate(('Left', 'Right')):
        norm_ax = fig.add_subplot(grid[0, hand])
        norm = np.linalg.norm(error[:, hand], axis=1)
        # Exact zeros are masked only because they cannot be placed on a log
        # axis. No positive residual is truncated or rounded to zero.
        norm_ax.semilogy(t, np.where(norm > 0, norm, np.nan), color=HAND_COLORS[hand],
                          label='Recorded estimate error to truth (offline)')
        norm_ax.axvspan(t[first_tail], t[-1], color='#888888', alpha=.09, label='Final 2 s')
        te = summary.get('ie_detection_times_s', [None, None])[hand]
        if te is not None:
            norm_ax.axvline(te, color='#444444', linestyle=':', linewidth=.9, label=f'IE/freeze at {te:g} s')
        norm_ax.set(title=f'{name}: recorded L2 error; final={norm[-1]:.4g}',
                    ylabel=r'$\|\hat\theta-\theta\|_2$', xlabel='Time (s)', xlim=(t[0], t[-1]))
        norm_ax.legend(loc='best', fontsize=7.1)
        component_ax = fig.add_subplot(grid[1, hand])
        drift_ax = fig.add_subplot(grid[2, hand])
        drift = theta[tail, hand]-theta[first_tail, hand]
        for j, color in enumerate(PARAMETER_COLORS):
            component_ax.plot(t, error[:, hand, j], color=color, linewidth=.85, label=rf'$\theta_{j+1}$')
            drift_ax.plot(t[tail], drift[:, j], color=color, linewidth=.8, label=rf'$\theta_{j+1}$')
        component_ax.axhline(0., color='#555555', linewidth=.6)
        component_ax.set(title=f'{name}: eight signed errors to offline truth',
                         ylabel='Multiplier error', xlabel='Time (s)', xlim=(t[0], t[-1]))
        component_ax.legend(loc='best', ncol=4, fontsize=7.5)
        drift_ax.axhline(0., color='#555555', linewidth=.6)
        net = np.linalg.norm(drift[-1])
        max_span = float(np.max(np.ptp(theta[tail, hand], axis=0)))
        drift_ax.set(title=(f'{name}: final-2-s drift; net L2={net:.3g}\n'
                            f'Max component peak-to-peak = {max_span:.3g}'),
                     ylabel=r'$\hat\theta_j(t)-\hat\theta_j(t_0)$', xlabel='Time (s)',
                     xlim=(t[first_tail], t[-1]))
        formatter = ticker.ScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_powerlimits((-2, 3)); drift_ax.yaxis.set_major_formatter(formatter)
    excitation = fig.add_subplot(grid[3, :])
    _raw_ie_axis(excitation, arrays, summary)
    _save(fig, out, '08_parameter_error_diagnostics', f'{title} | actual parameter error, drift and IE',
          'Complete original composite law retained. Offline truth is evaluation only. No exponential-to-truth curve is imposed; IE detection alone does not establish convergence. All samples retained.',
          layout_top=.935)


def plot_case(case_directory: Path, output: Path):
    """Write eight PNG/vector-PDF pairs for an original-method completed case."""
    case_directory, output = Path(case_directory), Path(output)
    summary = _read_json(case_directory/'summary.json')
    settings = summary.get('settings', {})
    gains = np.asarray(settings.get('adaptation_gain', []), dtype=float)
    if (settings.get('robust_mode') != 'paper_sign' or gains.shape != (8,)
            or not np.all(gains > 0) or float(settings.get('gamma', 0)) <= 0):
        raise ValueError('Expected original paper-sign results with a fixed positive eight-component diagonal gain and gamma>0')
    with np.load(case_directory/'trajectory.npz', allow_pickle=False) as recorded:
        missing = set(ARRAY_KEYS)-set(recorded.files)
        if missing:
            raise ValueError(f'Missing original-method plot arrays: {sorted(missing)}')
        arrays = {key:recorded[key] for key in ARRAY_KEYS}
    if len(arrays['t']) < 2 or not np.all(np.diff(arrays['t']) > 0):
        raise ValueError('Plotting requires at least two strictly increasing samples')
    if arrays['theta'].shape != (len(arrays['t']), 2, 8):
        raise ValueError('Expected eight recorded parameter estimates per arm')
    truth = _truth(case_directory, summary)
    metadata = _read_json(HERE/'models/scene_metadata.json')
    title = _title(summary, case_directory)
    output.mkdir(parents=True, exist_ok=True)
    _style()
    _trajectory(arrays, output, title)
    _errors(arrays, output, title)
    _transformed(arrays, output, title, settings)
    _torques(arrays, output, title, metadata)
    # The reused panel deliberately retains the original raw-Gram omega test.
    # Its estimate/truth/prior legend already distinguishes the three meanings.
    _learning(arrays, output, title, dict(summary, true_parameters_offline_only=truth.tolist()), metadata, {})
    _internal_force(arrays, output, title)
    _diagnostics(arrays, output, title)
    _parameter_error(arrays, output, title, summary, truth)
    return sorted(output.glob('0[1-8]_*.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE/'results_original_retrial')
    parser.add_argument('--case', nargs='*', help='Case names; default: all completed cases')
    parser.add_argument('--output', type=Path, help='Default: RESULTS/figures')
    args = parser.parse_args()
    root = args.results.resolve(); out = (args.output or root/'figures').resolve()
    cases = ([root/name for name in args.case] if args.case else
             sorted(p.parent for p in root.glob('*/source_manifest.json') if (p.parent/'trajectory.npz').exists()))
    if (root/'trajectory.npz').exists() and not args.case:
        cases = [root]
    if not cases:
        raise SystemExit(f'No completed original-method case under {root}')
    for directory in cases:
        figures = plot_case(directory, out/directory.name)
        print(f'{directory.name}: {len(figures)} PNG + {len(figures)} vector PDF in {out/directory.name}', flush=True)


if __name__ == '__main__':
    main()
