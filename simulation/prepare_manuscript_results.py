#!/usr/bin/env python3
"""Create compact manuscript figures directly from frozen comparison samples.

All line plots retain every sample in the displayed interval. No smoothing,
decimation, parameter fitting, or controller execution is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
METHODS = ("proposed", "traditional_bf", "linear_surface", "linear_reaching", "no_history", "external_ppc")
LABELS = dict(zip(METHODS, ("Proposed", "Log-BF", "Linear surface", "No dual-power", "No history", "External PPC")))
COLORS = dict(zip(METHODS, ("#1665a5", "#d17b0f", "#8e55ad", "#c34343", "#168d71", "#555555")))
STYLES = dict(zip(METHODS, ("-", "--", "-.", ":", "--", "-")))
PARAM_COLORS = ("#1665a5", "#d17b0f", "#168d71", "#c34343", "#8e55ad", "#725344", "#bf5687", "#777777")
OUTPUT_NAMES = ("cr7_tracking_comparison", "cr7_learning_comparison", "cr7_internal_wrench")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.2, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": .45,
        "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "path.simplify": False, "agg.path.chunksize": 10000,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def load_trial(root, scenario, method):
    directory = root / "main" / scenario / method
    summary = json.loads((directory / "summary.json").read_text())
    fields = ["t", "error", "upper", "lower", "actual_barrier_margin"]
    if scenario == "nominal":
        fields += ["internal_force", "desired_internal_force"]
        if method in ("proposed", "no_history"):
            fields += ["theta"]
    with np.load(directory / "trajectory.npz", allow_pickle=False) as raw:
        arrays = {name: raw[name] for name in fields}
    if summary["run_status"] != "completed":
        raise ValueError(f"Unexpected noncompleted manuscript source: {directory}")
    if len(arrays["t"]) == 0:
        raise ValueError(f"No saved samples in {directory}")
    return summary, arrays


def line(ax, t, y, method, *, width=.75):
    return ax.plot(t, y, color=COLORS[method], linestyle=STYLES[method],
                   linewidth=width, label=LABELS[method], alpha=.95)[0]


def method_legend(fig):
    handles = [Line2D([], [], color=COLORS[m], linestyle=STYLES[m], lw=1.05, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .998), ncol=6,
               columnspacing=1., handlelength=2., handletextpad=.6, frameon=False,
               fontsize=7)


def save(fig, output, name, description):
    if name in OUTPUT_NAMES:
        # Extra TeX memory preserves full-resolution vector paths.
        keys = ("extra_mem_top", "extra_mem_bot", "pool_size")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            os.environ.update(extra_mem_top="100000000", extra_mem_bot="100000000", pool_size="10000000")
            fig.savefig(output / f"{name}.pdf", backend="pgf")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        subprocess.run(["/usr/bin/pdftoppm", "-singlefile", "-r", "220", "-png",
                        str(output / f"{name}.pdf"), str(output / name)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        plt.close(fig)
        return
    fig.savefig(output / f"{name}.pdf", metadata={"Title": description, "Creator": "prepare_manuscript_results.py"})
    fig.savefig(output / f"{name}.png", dpi=220)
    plt.close(fig)


@matplotlib.rc_context({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                        "pgf.preamble": r"\usepackage{amsmath,amsfonts}\usepackage{times}",
                        "font.family": "serif"})
def tracking(nominal, nonmonotonic, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(left=.092, right=.985, bottom=.155, top=.85, wspace=.36, hspace=.70)
    method_legend(fig)
    for method in METHODS:
        a = nominal[method][1]
        line(axes[0, 0], a["t"], 1000 * np.linalg.norm(a["error"][:, :3], axis=1), method)
        line(axes[0, 1], a["t"], 180 / np.pi * np.linalg.norm(a["error"][:, 3:], axis=1), method)
        b = nonmonotonic[method][1]
        line(axes[1, 0], b["t"], b["error"][:, 0] * 1000, method)
        line(axes[1, 1], b["t"], b["error"][:, 4] * 180 / np.pi, method)
    axes[0, 0].set(yscale="log", ylabel=r"$\|e_p\|_2$ (mm)", title="(a) Nominal: position")
    axes[0, 1].set(yscale="log", ylabel=r"$\|e_{\mathrm{Euler}}\|_2$ (deg)", title="(b) Nominal: Euler orientation")
    axes[1, 0].set(ylabel=r"$x$-position error (mm)", title="(c) Nonmonotonic: position and common bounds")
    b = nonmonotonic["proposed"][1]
    for ax, component, numerator, denominator in ((axes[1, 0], 0, 1000, 1), (axes[1, 1], 4, 180, np.pi)):
        for key in ("upper", "lower"):
            ax.plot(b["t"], b[key][:, component] * numerator / denominator, color="black", ls="--", lw=.85)
        ax.axvspan(3, 7, color="#1665a5", alpha=.06)
        ax.text(.98, .93, "Dashed: common bounds", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.4)
    axes[1, 1].set(ylabel="Pitch error (deg)", title="(d) Nonmonotonic: pitch and common bounds")
    for ax in axes.flat:
        ax.set(xlim=(0, 24), xticks=[0, 6, 12, 18, 24], xlabel="Time (s)")
    save(fig, output, "cr7_tracking_comparison", "Nominal tracking norms and nonmonotonic x-position/pitch errors with common bounds; all saved samples")


@matplotlib.rc_context({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                        "pgf.preamble": r"\usepackage{amsmath,amsfonts}\usepackage{times}",
                        "font.family": "serif"})
def learning(nominal, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(left=.095, right=.985, bottom=.16, top=.85, wspace=.34, hspace=.85)
    norm_handles = [Line2D([], [], color=COLORS[m], linestyle=STYLES[m], lw=1.05, label=LABELS[m])
                    for m in ("proposed", "no_history")]
    fig.legend(handles=norm_handles, loc="upper center", bbox_to_anchor=(.515, .998),
               ncol=2, columnspacing=2.2, handlelength=2.7, frameon=False)
    for hand in range(2):
        for method in ("proposed", "no_history"):
            summary, a = nominal[method]
            truth = np.asarray(summary["true_parameters_offline_only"])
            line(axes[0, hand], a["t"], np.linalg.norm(a["theta"][:, hand] - truth[hand], axis=1), method, width=.85)
        summary, a = nominal["proposed"]
        truth = np.asarray(summary["true_parameters_offline_only"])
        for parameter in range(8):
            axes[1, hand].plot(a["t"], a["theta"][:, hand, parameter] - truth[hand, parameter],
                               color=PARAM_COLORS[parameter], linewidth=.7,
                               label=rf"$-\tilde\vartheta_{{i{parameter + 1}}}$")
        axes[1, hand].axhline(0, color="black", lw=.55)
        freeze = summary["ie_detection_times_s"][hand]
        for row in (0, 1):
            axes[row, hand].axvline(freeze, color="#777777", lw=.8, ls="--")
        arm = ("Left", "Right")[hand]
        axes[0, hand].set(title=f"({('a', 'b')[hand]}) {arm} arm: parameter error norm", ylabel=r"$\|\hat\vartheta_i-\vartheta_i\|_2$", ylim=(0, .41))
        axes[0, hand].text(freeze + .35, .025, "IE freeze: 6 s", fontsize=6.3, color="#666666")
        axes[1, hand].set(title=f"({('c', 'd')[hand]}) Proposed: {arm.lower()} signed errors", ylabel=r"$\hat\vartheta_{ij}-\vartheta_{ij}$", ylim=(-.255, .205))
    for ax in axes.flat:
        ax.set(xlim=(0, 24), xticks=[0, 6, 12, 18, 24], xlabel="Time (s)")
    save(fig, output, "cr7_learning_comparison", "Original composite learning and no-history ablation; signed multiplier errors; truth only for offline evaluation")


@matplotlib.rc_context({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                        "pgf.preamble": r"\usepackage{amsmath,amsfonts}\usepackage{times}",
                        "font.family": "serif"})
def wrench(nominal, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.8))
    fig.subplots_adjust(left=.09, right=.97, bottom=.30, top=.75, wspace=.35)
    method_legend(fig)
    for method in METHODS[1:] + METHODS[:1]:
        _, a = nominal[method]
        residual = a["internal_force"][:, 0] - a["desired_internal_force"][:, 0]
        line(axes[0], a["t"], np.linalg.norm(residual[:, :3], axis=1), method, width=.65)
        line(axes[1], a["t"], np.linalg.norm(residual[:, 3:], axis=1), method, width=.65)
    axes[0].set(ylabel=r"$\|[\tilde{\boldsymbol{f}}_L]_{1:3}\|_2$ (N)", title="(a) Nominal: left internal-force error")
    axes[0].set_yscale("symlog", linthresh=.2, linscale=.5)
    axes[1].set(ylabel=r"$\|[\tilde{\boldsymbol{f}}_L]_{4:6}\|_2$ (N m)", title="(b) Left Euler-conjugate moment error")
    axes[1].set_yscale("symlog", linthresh=.02, linscale=.5)
    for ax in axes:
        ax.set(xlim=(0, 24), ylim=(0, None), xticks=[0, 6, 12, 18, 24], xlabel="Time (s)")
        ax.text(.985, .055, "symlog", transform=ax.transAxes, ha="right", fontsize=6, color="#777777")
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
    save(fig, output, "cr7_internal_wrench", "Full-interval unsmoothed left internal wrench errors; force and Euler-conjugate moments separately")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT / "simulation" / "results_comparison")
    parser.add_argument("--output", type=Path, default=PROJECT / "manuscript_figures")
    args = parser.parse_args()
    root, output = args.results.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    style()
    nominal = {m: load_trial(root, "nominal", m) for m in METHODS}
    nonmonotonic = {m: load_trial(root, "nonmonotonic", m) for m in METHODS}
    tracking(nominal, nonmonotonic, output)
    print("Wrote cr7_tracking_comparison.pdf/png", flush=True)
    learning(nominal, output)
    print("Wrote cr7_learning_comparison.pdf/png", flush=True)
    wrench(nominal, output)
    print("Wrote cr7_internal_wrench.pdf/png", flush=True)
    inputs = {}
    for scenario in ("nominal", "nonmonotonic"):
        for method in METHODS:
            directory = root / "main" / scenario / method
            inputs[str(directory.relative_to(root))] = {
                "trajectory_sha256": sha256(directory / "trajectory.npz"),
                "summary_sha256": sha256(directory / "summary.json"),
                "samples": len((nominal if scenario == "nominal" else nonmonotonic)[method][1]["t"]),
            }
    manifest = {
        "script_sha256": sha256(Path(__file__)), "results_root": str(root),
        "method_labels": LABELS,
        "source_inputs": inputs, "all_displayed_interval_samples_retained": True,
        "smoothing": False, "line_simplification": False,
        "truth_use": "Offline signed parameter error and L2 evaluation only",
        "signed_error": "estimate minus truth",
        "parameter_legend": None,
        "common_margin": "min_j min((U_j-e_j)/U_j, (L_j+e_j)/L_j), L positive",
        "tracking_panels": {"a": "Nominal position error norm (mm)", "b": "Nominal Euler error norm (deg)",
                            "c": "Nonmonotonic signed x-position error and common bounds (mm)",
                            "d": "Nonmonotonic signed pitch error and common bounds (deg)"},
        "common_margin_displayed": False,
        "force": "left internal force minus left target; xyz norm separately from Euler-conjugate moment norm",
        "left_force_target_N": [0, -5, 0], "left_Euler_moment_target_Nm": [0, 0, 0],
        "learning_IE_freeze_s": nominal["proposed"][0]["ie_detection_times_s"],
        "output_dimensions_inches": {"cr7_tracking_comparison": [7.2, 3.1], "cr7_learning_comparison": [7.2, 3.1], "cr7_internal_wrench": [7.2, 1.8]},
        "method_legend_rows": 1,
        "local_magnification_views": False,
        "removed_views": ["Tracking margin inset and its interval highlight", "Final 0.1 s Fy detail", "Final 0.1 s roll-moment detail"],
        "external_nominal_minimum_common_margin": float(np.min(nominal["external_ppc"][1]["actual_barrier_margin"])),
        "outputs": {f"{name}.pdf": sha256(output / f"{name}.pdf") for name in OUTPUT_NAMES},
    }
    (output / "figure_data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
