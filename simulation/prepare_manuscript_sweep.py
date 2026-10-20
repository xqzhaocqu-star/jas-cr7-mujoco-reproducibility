#!/usr/bin/env python3
"""Paper-size initial-state study, recomputed from the unchanged raw results.

The physical entry criterion and method-specific S/xi diagnostic criteria are
kept separate. Censored times are never replaced by measured endpoint entries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparison.metrics import tracking_entry_metrics, sustained_entry_time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METHODS = ("proposed", "linear_surface", "linear_reaching")
LABELS = {"proposed": "Proposed", "linear_surface": "Linear surface",
          "linear_reaching": "No dual-power"}
COLORS = {"proposed": "#1665a5", "linear_surface": "#8e55ad", "linear_reaching": "#c34343"}
MARKERS = {"proposed": "o", "linear_surface": "s", "linear_reaching": "^"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_time(actual, expected):
    return (actual is None and expected is None) or (
        actual is not None and expected is not None and abs(actual - expected) < 1e-12)


def prepare_sweep(results, output_dir, *, dpi=600):
    results, output_dir = Path(results).resolve(), Path(output_dir).resolve()
    protocol_path, metrics_path = results / "protocol.json", results / "comparison_metrics.json"
    protocol = json.loads(protocol_path.read_text())
    metrics = {r["path"]: r for r in json.loads(metrics_path.read_text())["trials"]}
    scenarios = sorted((s for s in protocol["scenarios"] if s.get("study") == "initial_sweep"),
                       key=lambda s: s["initial_scale"])
    if len(scenarios) != 8:
        raise ValueError("This registered paper study must retain all eight initial states")
    settings = protocol["settings"]
    delta = float(settings["delta"])
    surface_threshold = float(protocol["evaluation"]["surface_component_abs"])
    hold_s = float(protocol["evaluation"]["hold_s"])
    position = float(protocol["evaluation"]["position_tolerance_m"])
    orientation = float(protocol["evaluation"]["orientation_coordinate_tolerance_rad"])
    rows = []
    for method in METHODS:
        for scenario in scenarios:
            relative = f'initial_sweep/{scenario["id"]}/{method}'
            directory = results / relative
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text())
            end = float(summary["requested_duration_s"])
            complete = summary["run_status"] == "completed"
            with np.load(directory / "trajectory.npz", allow_pickle=False) as raw:
                t, error = raw["t"], raw["error"]
                physical = tracking_entry_metrics(t, error, position, orientation,
                    hold_s=hold_s, requested_end_s=end, completed=complete)["simultaneous_entry_s"]
                surface = sustained_entry_time(t, np.max(np.abs(raw["surface"]), axis=(1, 2)) <= surface_threshold,
                    hold_s=hold_s, requested_end_s=end, completed=complete)
                layer = sustained_entry_time(t, np.max(np.abs(raw["xi"]), axis=(1, 2)) <= delta,
                    hold_s=hold_s, requested_end_s=end, completed=complete)
            aggregate = metrics[relative]
            expected = (aggregate["entry_sensitivity"]["1.0"]["simultaneous_entry_s"],
                        aggregate["phase_threshold_sensitivity"]["1.0"]["surface_entry_s"],
                        aggregate["phase_threshold_sensitivity"]["1.0"]["xi_layer_entry_s"])
            if not all(_same_time(x, y) for x, y in zip((physical, surface, layer), expected)):
                raise ValueError("Raw recomputation disagrees with the recorded aggregate: " + relative)
            rows.append(dict(method=method, scenario=scenario["id"], scale=float(scenario["initial_scale"]),
                initial_offset=scenario["offset"], completed=complete, requested_end_s=end,
                physical_entry_s=physical, surface_entry_s=surface, xi_entry_s=layer,
                trajectory=str(directory / "trajectory.npz"),
                trajectory_sha256=_sha256(directory / "trajectory.npz"),
                summary_sha256=_sha256(summary_path), aggregate_agrees=True))

    plt.rcParams.update({"font.family": "serif", "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsmath,amsfonts}\usepackage{times}", "font.size": 8., "axes.labelsize": 8.,
        "axes.titlesize": 9.5, "axes.titlepad": 3., "axes.labelpad": 2.,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "path.simplify": False, "savefig.facecolor": "white"})
    figure_size = [3.5, 2.0]
    fig = plt.figure(figsize=figure_size)
    grid = fig.add_gridspec(1, 1, left=.16, right=.98, bottom=.24, top=.82)
    left = fig.add_subplot(grid[0])
    counts = {}
    handles = []
    for i, method in enumerate(METHODS):
        selected = [r for r in rows if r["method"] == method]
        scale = np.asarray([r["scale"] for r in selected])
        y = np.asarray([r["physical_entry_s"] if r["physical_entry_s"] is not None else np.nan for r in selected])
        line, = left.plot(scale, y, marker=MARKERS[method], color=COLORS[method],
                          linestyle=("-", "--", "-.")[i], ms=4.4, lw=1.05, label=LABELS[method])
        handles.append(line)
        for row in selected:
            if row["physical_entry_s"] is None:
                left.scatter(row["scale"], row["requested_end_s"], marker="x", color=COLORS[method], s=27)
        counts[method] = [sum(r[key] is not None for r in selected)
                          for key in ("physical_entry_s", "surface_entry_s", "xi_entry_s")]
    ticks = np.asarray([s["initial_scale"] for s in scenarios])
    # Fraction glyphs avoid collisions at the true, closely spaced +/-1/4
    # and +/-1/2 ticks without changing the horizontal scale or text size.
    tick_labels = {-1.5: "-1.5", -1.: "-1", -.5: r"$-\tfrac{1}{2}$", -.25: r"$-\tfrac{1}{4}$",
                   .25: r"$+\tfrac{1}{4}$", .5: r"$+\tfrac{1}{2}$", 1.: "+1", 1.5: "+1.5"}
    left.set_xticks(ticks, [tick_labels[float(s)] for s in ticks])
    left.set(xlabel="Initial-error scale $s$", ylabel="Physical entry time (s)", xlim=(-1.67, 1.67), ylim=(0, 2.8))
    if any(r["physical_entry_s"] is None for r in rows):
        left.set_ylim(0, max(r["requested_end_s"] for r in rows) + .4)
    left.grid(axis="y", color="#dce1e5", linewidth=.55, zorder=0)
    left.set_axisbelow(True)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .99), ncol=3,
               frameon=False, fontsize=7., handlelength=1.5, columnspacing=.8,
               handletextpad=.4, borderaxespad=.15)
    # Tolerances and sustained-entry definition are stated in the manuscript
    # caption; the removed diagnostic counts remain in the text and provenance.
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "cr7_initial_condition_sweep"
    fig.savefig(stem.with_suffix(".pdf"), backend="pgf", facecolor="white")
    subprocess.run(["/usr/bin/pdftoppm", "-singlefile", "-r", str(dpi), "-png",
                    str(stem.with_suffix(".pdf")), str(stem)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    plt.close(fig)
    provenance = dict(purpose="Paper-size summary of the registered physical initial-condition sweep",
        script=str(Path(__file__).resolve()), script_sha256=_sha256(__file__),
        method_labels=LABELS, typography="PGF + pdfLaTeX; Times text and LaTeX math",
        results=str(results), protocol_sha256=_sha256(protocol_path), metrics_sha256=_sha256(metrics_path),
        figure_size_inches=figure_size, figure_size_mm=(np.asarray(figure_size)*25.4).tolist(), export_dpi=dpi,
        layout_revision="Single-column 3.5 x 2.0 in physical-entry plot; removed right diagnostic-count panel; counts retained in manuscript text and provenance; single-row abbreviated legend",
        base_text_size_pt=7., pdf_content="Vector lines, markers, axes and embedded fonts",
        panel_count=1, diagnostic_counts_displayed=False, legend_rows=1,
        signed_scale_order=ticks.tolist(), counts_order=["physical", "surface", "xi"],
        counts_by_method=counts, position_tolerance_m=position, orientation_coordinate_tolerance_rad=orientation,
        surface_component_threshold=surface_threshold, xi_component_threshold=delta, suffix_minimum_s=hold_s,
        definition="Entry is the first saved time in a valid suffix through the requested 8 s end, lasting >=2 s; None stays censored",
        unchanged_physics_and_raw=True, truth_parameters_read=False, rows=rows,
        output_png=str(stem.with_suffix(".png")), output_pdf=str(stem.with_suffix(".pdf")),
        output_png_sha256=_sha256(stem.with_suffix(".png")),
        output_pdf_sha256=_sha256(stem.with_suffix(".pdf")),
        caption_suggestion=("Physical-tolerance entry times for eight signed scales of one fixed "
          "six-coordinate initial-error direction. Position/Euler tolerances are 0.25 mm/0.05 deg, "
          "sustained through 8 s for at least 2 s. All controller settings remain fixed."))
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results_comparison")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "manuscript_figures")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    info = prepare_sweep(args.results, args.output_dir, dpi=args.dpi)
    print(json.dumps({k: info[k] for k in ("output_png", "output_pdf", "counts_by_method")}, indent=2))


if __name__ == "__main__":
    main()
