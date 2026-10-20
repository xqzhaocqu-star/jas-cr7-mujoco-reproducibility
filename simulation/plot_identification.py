#!/usr/bin/env python3
"""Plot the declared measured-force identification extension, not original JAS.

Eight PNG/vector-PDF pairs per case are written under RESULTS/figures/CASE.
Offline truths are loaded only from completed result evaluation records.
Every recorded sample is retained, including torque extrema and error floors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from plot_results import (CASE_TITLES, HAND_COLORS, HERE, _diagnostics, _errors,
                          _internal_force, _read_json, _save, _style, _time_axes,
                          _torques, _trajectory, _transformed)

PLOT_ARRAYS = ("t", "object_pose", "reference", "error", "upper", "lower",
               "surface", "xi", "torque_applied", "torque_command", "theta",
               "normalized_lambda_min", "internal_force", "desired_internal_force",
               "weld_residual", "barrier_margin", "sigma_min", "contact_count")
PARAMETER_NAMES = [f"Link {j} density" for j in range(1, 7)] + ["Common viscous scale", "Payload density"]
PARAMETER_SHORT = ["L1", "L2", "L3", "L4", "L5", "L6", "visc.", "payload"]


def _extension_title(summary, directory):
    settings = summary["settings"]
    case = CASE_TITLES.get(summary.get("case", directory.name), directory.name.replace("_", " ").title())
    return (f"MEASURED-FORCE IDENTIFICATION EXTENSION | {case} | "
            f"extra sensing + new history update; r={float(settings['identification_rate']):g} s⁻¹")


def _load_evaluation(case_directory, summary):
    path = case_directory / "learning_state.npz"
    state = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            for key in ("true_parameters_offline_only", "data_target", "theta_at_freeze"):
                if key in data.files:
                    state[key] = data[key]
    for key in ("true_parameters_offline_only", "data_target"):
        if key not in state and key in summary:
            state[key] = np.asarray(summary[key], dtype=float)
        if key not in state or state[key].shape != (2, 8):
            raise ValueError(f"Completed evaluation must provide {key} with shape (2,8)")
    return state


def _bounds(metadata, settings):
    parameterization = metadata.get("parameterization", {})
    lower = np.asarray(parameterization.get("local_prior_lower", [settings["prior_lower"]] * 8))
    upper = np.asarray(parameterization.get("local_prior_upper", [settings["prior_upper"]] * 8))
    return np.broadcast_to(lower, (8,)), np.broadcast_to(upper, (8,))


def _learning(a, out, title, summary, metadata, evaluation):
    t, settings = a["t"], summary["settings"]
    lower, upper = _bounds(metadata, settings)
    truth = evaluation["true_parameters_offline_only"]
    threshold = float(settings["identification_threshold"])
    detection = summary.get("ie_detection_times_s", [None, None])
    fig = plt.figure(figsize=(11.5, 12.6))
    grid = fig.add_gridspec(5, 2, height_ratios=(1, 1, 1, 1, 1.25))
    axes = []
    for j in range(8):
        ax = fig.add_subplot(grid[j // 2, j % 2]); axes.append(ax)
        ax.axhspan(lower[j], upper[j], color="#777777", alpha=.07)
        ax.axhline(lower[j], color="#555555", linestyle=":", linewidth=.8)
        ax.axhline(upper[j], color="#555555", linestyle=":", linewidth=.8, label="Prior / projection limits")
        for hand in range(2):
            name, color = ("Left", HAND_COLORS[0]) if hand == 0 else ("Right", HAND_COLORS[1])
            ax.plot(t, a["theta"][:, hand, j], color=color, label=f"{name} estimate")
            ax.axhline(truth[hand, j], color=color, linestyle="--", linewidth=.85,
                       label=f"{name} truth (offline)")
        pad = .06*(upper[j]-lower[j])
        ax.set(title=rf"$\theta_{j+1}$: {PARAMETER_NAMES[j]}", ylabel="Multiplier",
               ylim=(lower[j]-pad, upper[j]+pad))
        ticks = ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(lower[j], upper[j])
        ax.set_yticks(np.r_[lower[j], ticks[(ticks > lower[j]+1e-12) & (ticks < upper[j]-1e-12)], upper[j]])
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()
            order = list(range(1, len(handles))) + [0]
            fig.legend([handles[k] for k in order], [labels[k] for k in order], ncol=3,
                       loc="upper center", bbox_to_anchor=(.5, .935), frameon=False)
    ax = fig.add_subplot(grid[4, :]); axes.append(ax)
    descriptions = []
    for hand in range(2):
        name = "Left" if hand == 0 else "Right"
        ax.plot(t, a["normalized_lambda_min"][:, hand], color=HAND_COLORS[hand],
                label=f"{name} measured normalized λmin")
        te = detection[hand]
        descriptions.append(f"{name}: " + (f"frozen at {te:g} s" if te is not None else "not frozen"))
        if te is not None:
            ax.axvline(te, color=HAND_COLORS[hand], linestyle=":", linewidth=1.)
    ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.,
               label=f"Identification threshold = {threshold:g}")
    linthresh = threshold*.05
    ax.set_yscale("symlog", linthresh=linthresh)
    low = float(np.min(a["normalized_lambda_min"]))
    high = max(threshold, float(np.max(a["normalized_lambda_min"])))
    ax.set_ylim(min(low, 0.)-.05*linthresh, 1.5*high)
    exponents = np.arange(int(np.ceil(np.log10(linthresh))), int(np.floor(np.log10(1.5*high)))+1)
    positive = 10.**exponents
    negative = -positive[positive <= -low][::-1]
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.r_[negative, 0., positive]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda value, _: "0" if value == 0 else rf"${'-' if value < 0 else ''}10^{{{int(np.round(np.log10(abs(value))))}}}$"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set(title="; ".join(descriptions), ylabel=r"$\lambda_{\min}(D^{-1/2}GD^{-1/2})$")
    ax.legend(loc="best", ncol=3)
    _time_axes(axes, t)
    _save(fig, out, "05_learning", f"{title} | eight estimates and normalized excitation",
          "Dashed colored lines are offline truths; gray dotted lines are priors. The lower panel uses the new normalized identification threshold, not the original JAS ω.",
          layout_top=.905)


def _positive_for_log(values):
    """Mask exact zeros; never replace a recorded zero by an apparent floor."""
    values = np.asarray(values)
    return np.where(values > 0, values, np.nan)


def _exponential(a, out, title, summary, evaluation):
    t, theta = a["t"], a["theta"]
    truth, target = evaluation["true_parameters_offline_only"], evaluation["data_target"]
    rate = float(summary["settings"]["identification_rate"])
    detection = summary.get("ie_detection_times_s", [None, None])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.9), gridspec_kw={"height_ratios": [1.45, 1.]})
    for hand in range(2):
        ax, bars = axes[:, hand]
        color = HAND_COLORS[hand]
        name = "Left" if hand == 0 else "Right"
        truth_error = np.linalg.norm(theta[:, hand]-truth[hand], axis=1)
        ax.semilogy(t, _positive_for_log(truth_error), color=color,
                    linewidth=1.2, label="Recorded error to truth (offline)")
        te = detection[hand]
        finite_target = bool(np.all(np.isfinite(target[hand])))
        floor = np.linalg.norm(target[hand]-truth[hand]) if finite_target else np.nan
        if te is not None and finite_target:
            post = t >= te-1e-12
            target_error = np.linalg.norm(theta[post, hand]-target[hand], axis=1)
            at_freeze = evaluation.get("theta_at_freeze")
            if at_freeze is None or not np.all(np.isfinite(at_freeze[hand])):
                freeze_theta = theta[np.argmin(np.abs(t-te)), hand]
            else:
                freeze_theta = at_freeze[hand]
            initial_distance = np.linalg.norm(freeze_theta-target[hand])
            theory = initial_distance*np.exp(-rate*(t[post]-te))
            upper, lower = theory+floor, np.abs(theory-floor)
            ax.fill_between(t[post], _positive_for_log(lower), upper, color=color, alpha=.09,
                            label="Truth-error triangle envelope")
            ax.semilogy(t[post], _positive_for_log(upper), color=color, linestyle="-.",
                        linewidth=.9, label="Truth upper envelope: decay + floor")
            ax.semilogy(t[post], _positive_for_log(target_error), color="#1b9e77",
                        linewidth=1.3, label="Recorded error to frozen data target")
            ax.semilogy(t[post], _positive_for_log(theory), color="#333333", linestyle="--",
                        linewidth=.9, label=rf"Target decay $e^{{-{rate:g}(t-T_e)}}$")
            if floor > 0:
                ax.axhline(floor, color="#b32424", linestyle=":", linewidth=1.1,
                           label=f"Offline truth floor = {floor:.3g}")
            ax.axvline(te, color="#666666", linestyle=":", linewidth=.8)
            ax.text(.02, .025, f"$T_e={te:g}$ s", transform=ax.transAxes, fontsize=8, color="#444444")
        else:
            ax.text(.03, .05, "No valid frozen data target", transform=ax.transAxes, color="#b32424")
        ax.set(title=f"{name} arm: exponential transient and retained truth floor",
               ylabel=r"Parameter error $\|\cdot\|_2$", xlabel="Time (s)", xlim=(t[0], t[-1]))
        ax.legend(loc="upper right", fontsize=7.1, framealpha=.94)
        j = np.arange(8)
        final_error = theta[-1, hand]-truth[hand]
        bars.bar(j, final_error, color=color, alpha=.78, width=.68, label="Final estimate − truth")
        if finite_target:
            bars.plot(j, target[hand]-truth[hand], color="#222222", marker="_", markersize=11,
                      linestyle="none", markeredgewidth=1.4, label="Frozen data target − truth")
        bars.axhline(0., color="#666666", linewidth=.6)
        bars.set(xticks=j, xticklabels=PARAMETER_SHORT, ylabel="Signed multiplier error",
                 title=f"{name} final eight errors: L2={np.linalg.norm(final_error):.3g}")
        formatter = ticker.ScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_powerlimits((-2, 3)); bars.yaxis.set_major_formatter(formatter)
        bars.tick_params(axis="x", labelsize=8)
        bars.legend(loc="best", fontsize=7.2)
    _save(fig, out, "08_exponential_identification", f"{title} | exponential learning and numerical floor",
          "The new history law converges exponentially to a measured-data target. Its offline error to truth remains visible; the target is not a true-parameter oracle. Exact zeros are omitted on log axes.",
          layout_top=.925)


def plot_case(case_directory: Path, output: Path):
    """Write exactly eight extension-specific figure pairs for a completed case."""
    case_directory, output = Path(case_directory), Path(output)
    summary = _read_json(case_directory/"summary.json")
    settings = summary.get("settings", {})
    for key in ("identification_rate", "identification_threshold"):
        if key not in settings:
            raise ValueError(f"This plotter requires measured-force identification results: missing {key}")
    with np.load(case_directory/"trajectory.npz", allow_pickle=False) as data:
        missing = [key for key in PLOT_ARRAYS if key not in data.files]
        if missing:
            raise ValueError(f"Missing recorded extension arrays: {missing}")
        arrays = {key: data[key] for key in PLOT_ARRAYS}
    if len(arrays["t"]) < 2 or not np.all(np.diff(arrays["t"]) > 0):
        raise ValueError("At least two strictly increasing recorded timestamps are required")
    if arrays["theta"].shape != (len(arrays["t"]), 2, 8):
        raise ValueError("Expected eight parameter estimates per arm")
    evaluation = _load_evaluation(case_directory, summary)
    metadata = _read_json(HERE/"models/scene_metadata.json")
    title = _extension_title(summary, case_directory)
    output.mkdir(parents=True, exist_ok=True)
    _style()
    _trajectory(arrays, output, title)
    _errors(arrays, output, title)
    _transformed(arrays, output, title, settings)
    _torques(arrays, output, title, metadata)
    _learning(arrays, output, title, summary, metadata, evaluation)
    _internal_force(arrays, output, title)
    _diagnostics(arrays, output, title)
    _exponential(arrays, output, title, summary, evaluation)
    return sorted(output.glob("0[1-8]_*.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results_exponential")
    parser.add_argument("--case", nargs="*", help="Case names; default is every completed case")
    parser.add_argument("--output", type=Path, help="Default: RESULTS/figures")
    args = parser.parse_args()
    root = args.results.resolve(); output = (args.output or root/"figures").resolve()
    cases = [root/c for c in args.case] if args.case else sorted(p.parent for p in root.glob("*/trajectory.npz"))
    if (root/"trajectory.npz").exists() and not args.case:
        cases = [root]
    if not cases:
        raise SystemExit(f"No completed identification trajectory under {root}")
    for directory in cases:
        files = plot_case(directory, output/directory.name)
        print(f"{directory.name}: {len(files)} PNG + {len(files)} vector PDF in {output/directory.name}", flush=True)


if __name__ == "__main__":
    main()
