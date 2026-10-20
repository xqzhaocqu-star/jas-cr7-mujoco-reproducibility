#!/usr/bin/env python3
"""Plot recorded JAS experiments without resampling any time-series data.

PNG previews and vector PDFs are written to RESULTS/figures/CASE.  The seven
figure types describe one controller under different JAS task conditions; they
do not introduce comparison controllers. Plant truth is read for offline plots
only and is never supplied to a controller by this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

HERE = Path(__file__).resolve().parent
HAND_COLORS = ("#2166ac", "#d66024")
COMPONENT_COLORS = ("#2166ac", "#1b9e77", "#c74343", "#8455a5", "#cc8b19", "#626262")
CASE_TITLES = {"nominal": "Nominal task", "nonmonotonic": "Nonmonotonic performance bounds",
               "large_initial_error": "Larger initial error"}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "figure.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": .55,
                         "lines.linewidth": 1., "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         # Keep every sampled torque extremum in both output formats.
                         "path.simplify": False, "agg.path.chunksize": 10000})


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _mode_label(summary):
    settings = summary.get("settings", {})
    mode = settings.get("robust_mode")
    if mode == "boundary_layer":
        return f"Boundary-layer engineering variant (ε={float(settings['robust_width']):g})"
    if mode == "paper_sign":
        return "Paper sign feedback"
    return ""


def _save(fig, out, stem, title, note=None, *, layout_top=None):
    title = "\n".join(textwrap.wrap(title, width=115, break_long_words=False))
    fig.suptitle(title, y=.995)
    if note:
        fig.text(.5, .007, note, ha="center", va="bottom", fontsize=8, color="#555555")
    top = (.945 if "\n" in title else .975) if layout_top is None else layout_top
    fig.tight_layout(rect=(0, .025 if note else .005, 1, top))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=180)
    plt.close(fig)


def _time_axes(axes, time):
    for ax in np.asarray(axes).ravel():
        ax.set_xlim(time[0], time[-1] if time[-1] > time[0] else time[0] + .1)
        ax.set_xlabel("Time (s)")


def _trajectory(a, out, title):
    actual, reference = a["object_pose"][:, :3] * 1000, a["reference"][:, :3] * 1000
    fig = plt.figure(figsize=(11.5, 6.3))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1))
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    ax.plot(*actual.T, color=HAND_COLORS[0], label="Physical payload COM")
    ax.plot(*reference.T, color=HAND_COLORS[1], linestyle="--", label="Reference COM")
    ax.scatter(*actual[0], color="#1b9e77", marker="o", s=25, label="Actual start")
    ax.scatter(*actual[-1], color="#333333", marker="s", s=22, label="Actual end")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    span = np.ptp(np.vstack((actual, reference)), axis=0)
    ax.set_box_aspect(np.maximum(span, max(span.max() * .3, 1.)))
    ax.view_init(elev=24, azim=-54)
    ax.legend(loc="upper left", fontsize=8)
    for ax, dims in [(fig.add_subplot(grid[0, 1]), (0, 1)),
                     (fig.add_subplot(grid[1, 1]), (1, 2))]:
        x, y = dims
        ax.plot(actual[:, x], actual[:, y], color=HAND_COLORS[0])
        ax.plot(reference[:, x], reference[:, y], color=HAND_COLORS[1], linestyle="--")
        ax.scatter(actual[0, x], actual[0, y], color="#1b9e77", s=15)
        ax.scatter(actual[-1, x], actual[-1, y], color="#333333", marker="s", s=15)
        ax.set(xlabel=f"{'xyz'[x]} (mm)", ylabel=f"{'xyz'[y]} (mm)")
        ax.set_aspect("equal", adjustable="datalim")
    _save(fig, out, "01_trajectory", f"{title} | physical payload trajectory")


def _errors(a, out, title):
    t = a["t"]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 8.1))
    names = ("x", "y", "z", "roll", "pitch", "yaw")
    for j in range(6):
        ax = axes[j % 3, j // 3]
        scale, unit = (1000., "mm") if j < 3 else (180 / np.pi, "deg")
        ax.plot(t, a["error"][:, j] * scale, color=HAND_COLORS[0], label="Physical error")
        valid = ((t >= .02) & np.isfinite(a["upper"][:, j]) & np.isfinite(a["lower"][:, j]))
        upper = np.where(valid, a["upper"][:, j] * scale, np.nan)
        lower = np.where(valid, a["lower"][:, j] * scale, np.nan)
        ax.plot(t, upper, "--", color=HAND_COLORS[1], label="Asymmetric bounds")
        ax.plot(t, lower, "--", color=HAND_COLORS[1])
        ax.fill_between(t, lower, upper, color="#d66024", alpha=.055)
        ax.axhline(0., color="#777777", linewidth=.55)
        ax.set(title=f"{names[j]} error", ylabel=f"Error ({unit})")
        # Display the small sampled steady-state residual without changing the
        # main physical error/bound scale or discarding transient samples.
        tail = t >= max(t[0], t[-1] - 2.)
        inset = ax.inset_axes([.56, .10, .40, .28])
        inset.plot(t[tail], a["error"][tail, j] * scale, color=HAND_COLORS[0], linewidth=.55)
        inset.set_xlim(t[tail][0], t[-1])
        inset.set_title("Final 2 s: error zoom", fontsize=6.5, pad=2)
        inset.tick_params(labelsize=6, pad=1, length=2)
        inset.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        inset.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        formatter = ticker.ScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_powerlimits((-2, 3))
        inset.yaxis.set_major_formatter(formatter)
        inset.yaxis.get_offset_text().set_fontsize(6)
    axes[0, 0].legend(loc="best")
    _time_axes(axes, t)
    _save(fig, out, "02_errors", f"{title} | six physical errors and performance bounds",
          "Bounds start at 0.02 s; initially infinite bounds are omitted. Insets show the final 2 s of physical error. Every sample is retained.")


def _transformed(a, out, title, settings):
    t, delta = a["t"], float(settings.get("delta", .0005))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    for hand in range(2):
        for j, c in enumerate(COMPONENT_COLORS):
            axes[0, hand].plot(t, a["surface"][:, hand, j], color=c, label=f"$S_{j+1}$")
            axes[1, hand].plot(t, a["xi"][:, hand, j], color=c, label=rf"$\xi_{j+1}$")
        axes[0, hand].set(title=f"{'Left' if hand == 0 else 'Right'} arm: surface components", ylabel="$S_j$")
        axes[0, hand].set_yscale("symlog", linthresh=delta)
        axes[1, hand].axhline(delta, color="#222222", linestyle="--", linewidth=.9)
        axes[1, hand].axhline(-delta, color="#222222", linestyle="--", linewidth=.9)
        axes[1, hand].axhspan(-delta, delta, color="#777777", alpha=.10)
        axes[1, hand].set(ylabel=rf"$\xi_j$; $\delta={delta:g}$", title="Transformed error and practical region")
        axes[1, hand].set_yscale("symlog", linthresh=delta)
        axes[0, hand].legend(ncol=3, loc="upper right")
        axes[1, hand].legend(ncol=3, loc="upper right")
    _time_axes(axes, t)
    _save(fig, out, "03_transformed_states", f"{title} | JAS surface and transformed error",
          "Symmetric log scales show transients and the practical region together; dashed lines mark ±δ for ξ only.")


def _torques(a, out, title, metadata):
    t = a["t"]
    limits = np.asarray(metadata.get("torque_limits_Nm", [194, 194, 102, 66, 66, 29]))
    fig, axes = plt.subplots(6, 2, figsize=(11.5, 11.8))
    for hand in range(2):
        for j in range(6):
            k, ax = 6 * hand + j, axes[j, hand]
            applied, command = a["torque_applied"][:, k], a["torque_command"][:, k]
            ax.plot(t, applied, color=HAND_COLORS[hand], linewidth=.65, label="Applied")
            if np.any(np.abs(command - applied) > 1e-10):
                ax.plot(t, command, color="#666666", alpha=.7, linewidth=.55, label="Commanded")
            ax.axhline(limits[j], color="#b32424", linestyle="--", linewidth=.8)
            ax.axhline(-limits[j], color="#b32424", linestyle="--", linewidth=.8, label="Physical limits")
            peak = max(limits[j], np.max(np.abs(command)), np.max(np.abs(applied)))
            ax.set_ylim(-1.08 * peak, 1.08 * peak)
            ax.set(title=f"{'Left' if hand == 0 else 'Right'} joint {j + 1}", ylabel="Torque (N m)")
            if j == 0:
                ax.legend(loc="upper right", ncol=2)
    _time_axes(axes, t)
    _save(fig, out, "04_torques", f"{title} | twelve motor torques and source actuator limits",
          "All controller-rate samples are drawn. No time-series downsampling or path simplification is applied.")


def _learning(a, out, title, summary, metadata, truth):
    t = a["t"]
    settings = summary.get("settings", {})
    parameterization = metadata.get("parameterization", {})
    lower = np.asarray(parameterization.get("local_prior_lower", [.85] * 8))
    upper = np.asarray(parameterization.get("local_prior_upper", [1.15] * 8))
    true_theta = summary.get("true_parameters_offline_only")
    if true_theta is None and truth:
        true_theta = [truth[n + "_density_multipliers"] + [truth[n + "_viscous_multiplier"], truth["payload_density_multiplier"]]
                      for n in ("left", "right")]
    fig = plt.figure(figsize=(11.5, 12.0))
    gs = fig.add_gridspec(5, 2, height_ratios=(1, 1, 1, 1, 1.25))
    axes = []
    names = [f"Link {i} density" for i in range(1, 7)] + ["Common viscous scale", "Payload density"]
    for j in range(8):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        axes.append(ax)
        ax.axhspan(lower[j], upper[j], color="#777777", alpha=.07)
        ax.axhline(lower[j], color="#555555", linestyle=":", linewidth=.9)
        ax.axhline(upper[j], color="#555555", linestyle=":", linewidth=.9,
                   label="Projection bounds (gray)")
        for hand in range(2):
            ax.plot(t, a["theta"][:, hand, j], color=HAND_COLORS[hand], label=f"{'Left' if hand == 0 else 'Right'} estimate")
            if true_theta is not None:
                ax.axhline(true_theta[hand][j], color=HAND_COLORS[hand], linestyle="--", linewidth=.8,
                           label=f"{'Left' if hand == 0 else 'Right'} true value (offline)")
        pad = .06 * (upper[j] - lower[j])
        ax.set(title=rf"$\theta_{j+1}$: {names[j]}", ylabel="Multiplier", ylim=(lower[j] - pad, upper[j] + pad))
        interior_ticks = ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(lower[j], upper[j])
        interior_ticks = interior_ticks[(interior_ticks > lower[j] + 1e-12)
                                        & (interior_ticks < upper[j] - 1e-12)]
        ax.set_yticks(np.r_[lower[j], interior_ticks, upper[j]])
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Put the bound key last; colored dashed lines are two truths,
            # not the lower/upper edges of a confidence or feasible interval.
            order = list(range(1, len(handles))) + [0]
            fig.legend([handles[k] for k in order], [labels[k] for k in order],
                       loc="upper center", bbox_to_anchor=(.5, .955),
                       ncol=3, frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[4, :])
    axes.append(ax)
    omega = float(settings.get("omega", 1e-6))
    detection = summary.get("ie_detection_times_s", [None, None])
    descriptions = []
    for hand in range(2):
        name = "Left" if hand == 0 else "Right"
        ax.plot(t, a["lambda_min"][:, hand], color=HAND_COLORS[hand], label=f"{name} measured λmin")
        detected = detection[hand]
        descriptions.append(f"{name}: " + (f"IE detected at {detected:.3f} s" if detected is not None else "IE not detected"))
        if detected is not None:
            ax.axvline(detected, color=HAND_COLORS[hand], linestyle=":", linewidth=1.2)
    ax.axhline(omega, color="#222222", linestyle="--", label=rf"IE threshold $\omega={omega:g}$")
    linthresh = max(omega * .05, 1e-14)
    ax.set_yscale("symlog", linthresh=linthresh)
    low, high = float(np.min(a["lambda_min"])), max(float(np.max(a["lambda_min"])), omega)
    ax.set_ylim(min(low, 0.) - linthresh * .1, high * 1.8)
    exponents = np.arange(int(np.ceil(np.log10(linthresh))), int(np.floor(np.log10(high * 1.8))) + 1)
    if len(exponents) > 6:
        exponents = exponents[np.linspace(0, len(exponents) - 1, 6).round().astype(int)]
    positive = 10. ** exponents
    negative = -positive[positive <= -low][::-1]
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.r_[negative, 0., positive]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: "0" if v == 0 else rf"${'-' if v < 0 else ''}10^{{{int(np.round(np.log10(abs(v))))}}}$"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set(ylabel=r"$\lambda_{\min}(\check{\Theta})$",
           title="; ".join(descriptions))
    ax.legend(loc="best", ncol=3)
    _time_axes(axes, t)
    _save(fig, out, "05_learning", f"{title} | parameter estimates, priors, and measured excitation",
          "Colored dashes: offline truths, not bounds. Gray dotted lines: projection limits. IE threshold: a lower test, not an upper bound.",
          layout_top=.925)


def _internal_force(a, out, title):
    t = a["t"]
    error = a["internal_force"] - a["desired_internal_force"]
    tail = t >= max(t[0], t[-1] - 2.)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for hand in range(2):
        rms = np.sqrt(np.mean(error[tail, hand] ** 2, axis=0))
        # Draw the largest raw component first so its dense switching band does
        # not cover the smaller components. Every original sample remains.
        force_order = np.argsort(np.max(np.abs(error[:, hand, :3]), axis=0))[::-1]
        moment_order = np.argsort(np.max(np.abs(error[:, hand, 3:]), axis=0))[::-1]
        for j in force_order:
            axes[0, hand].plot(t, error[:, hand, j], color=COMPONENT_COLORS[j], label=f"{'xyz'[j]}: {rms[j]:.3g}")
        for j in moment_order:
            axes[1, hand].plot(t, error[:, hand, j + 3], color=COMPONENT_COLORS[j], label=f"{('roll', 'pitch', 'yaw')[j]}: {rms[j+3]:.3g}")
        axes[0, hand].set(title=f"{'Left' if hand == 0 else 'Right'} arm: internal force error", ylabel="Force error (N)")
        axes[1, hand].set(title="Internal generalized moment error", ylabel="Moment error (N m)")
        axes[0, hand].legend(loc="best", ncol=3, title="Final 2 s RMS (N)", title_fontsize=8)
        axes[1, hand].legend(loc="best", ncol=3, title="Final 2 s RMS (N m)", title_fontsize=8)
        for ax in axes[:, hand]:
            ax.axhline(0., color="#777777", linewidth=.55)
    _time_axes(axes, t)
    _save(fig, out, "06_internal_force", f"{title} | internal force and moment tracking errors",
          "Force is in N. Moments are in N m and conjugate to Euler roll/pitch/yaw; they are not Cartesian torque components.")


def _diagnostics(a, out, title):
    t = a["t"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.1))
    for hand in range(2):
        name, color = ("Left", HAND_COLORS[0]) if hand == 0 else ("Right", HAND_COLORS[1])
        residual = a["weld_residual"][:, 6 * hand:6 * hand + 6]
        axes[0, 0].plot(t, np.max(np.abs(residual[:, :3]), axis=1) * 1000., color=color, label=name)
        axes[0, 1].plot(t, np.max(np.abs(residual[:, 3:6]), axis=1), color=color, label=name)
        margin = a["barrier_margin"][:, hand]
        if margin.ndim > 1:
            margin = np.min(margin, axis=-1)
        axes[1, 0].plot(t, margin, color=color, label=name)
        axes[1, 1].plot(t, a["sigma_min"][:, hand], color=color, label=name)
    axes[0, 0].set(title="Weld: largest translation residual", ylabel="Residual (mm)")
    axes[0, 1].set(title="Weld: largest scaled quaternion residual", ylabel="MuJoCo rotational constraint residual")
    axes[1, 0].axhline(0., color="#b32424", linestyle="--", linewidth=.8)
    axes[1, 0].set(title="Normalized barrier gap", ylabel="Minimum component gap / ℓ")
    axes[1, 1].axhline(.04, color="#777777", linestyle=":", linewidth=.8, label="0.04 diagnostic level")
    axes[1, 1].set(title="Object-pose Jacobian conditioning", ylabel=r"$\sigma_{\min}(J)$")
    for ax in axes.ravel():
        ax.legend(loc="best")
    _time_axes(axes, t)
    contacts = int(np.max(a.get("contact_count", np.array([0]))))
    _save(fig, out, "07_diagnostics", f"{title} | rigid-grasp, barrier, and kinematic diagnostics",
          f"Rotational weld rows include the MJCF torquescale; they are not rotation angles. Maximum recorded contact count: {contacts}.")


def plot_case(case_directory: Path, output: Path):
    """Produce seven PNG/PDF figure pairs from one completed trajectory NPZ."""
    case_directory, output = Path(case_directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(case_directory / "trajectory.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if len(arrays["t"]) < 2 or not np.all(np.diff(arrays["t"]) > 0):
        raise ValueError("Plotting requires at least two strictly increasing recorded timestamps")
    summary = _read_json(case_directory / "summary.json")
    metadata = _read_json(HERE / "models" / "scene_metadata.json")
    truth = _read_json(HERE / "models" / "plant_truth.json")
    title = CASE_TITLES.get(summary.get("case", case_directory.name), case_directory.name.replace("_", " ").title())
    mode_label = _mode_label(summary)
    if mode_label:
        title += f" | {mode_label}"
    _style()
    _trajectory(arrays, output, title)
    _errors(arrays, output, title)
    _transformed(arrays, output, title, summary.get("settings", {}))
    _torques(arrays, output, title, metadata)
    _learning(arrays, output, title, summary, metadata, truth)
    _internal_force(arrays, output, title)
    _diagnostics(arrays, output, title)
    return sorted(output.glob("*.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--case", nargs="*", help="case names, default: every completed case")
    parser.add_argument("--output", type=Path, help="default: RESULTS/figures")
    args = parser.parse_args()
    results, out = args.results.resolve(), (args.output or args.results / "figures").resolve()
    cases = [results / c for c in args.case] if args.case else sorted(p.parent for p in results.glob("*/trajectory.npz"))
    if (results / "trajectory.npz").exists() and not args.case:
        cases = [results]
    if not cases:
        raise SystemExit(f"No completed trajectory.npz files found under {results}")
    for directory in cases:
        files = plot_case(directory, out / directory.name)
        print(f"{directory.name}: {len(files)} PNG + {len(files)} vector PDF figures in {out / directory.name}", flush=True)


if __name__ == "__main__":
    main()
