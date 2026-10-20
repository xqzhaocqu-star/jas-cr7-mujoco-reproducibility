#!/usr/bin/env python3
"""Compare raw paper_sign and boundary_layer JAS rollouts after validation.

Writes a Chinese review, machine-readable metrics, and two PNG/vector-PDF
figure pairs. All force statistics and time traces use original control ticks.
The boundary layer is identified as an engineering change to the paper law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
CASES = ("nominal", "nonmonotonic", "large_initial_error")
TITLES = ("Nominal task", "Nonmonotonic bounds", "Larger initial error")
COMPONENTS = ("Fx", "Fy", "Fz", "Euler_roll_moment", "Euler_pitch_moment", "Euler_yaw_moment")
UNITS = ("N", "N", "N", "N m", "N m", "N m")
COLORS = {"paper_sign": "#c45432", "boundary_layer": "#2166ac"}


def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation(path):
    validation = _json(path / "validation.json")
    if validation is None:
        return {"status": "MISSING", "passed": None, "check_count": 0, "failures": []}
    checks = validation.get("checks", {})
    failures = [name for name, check in checks.items() if not check.get("passed", False)]
    return {"status": "PASS" if validation.get("passed", False) else "FAIL",
            "passed": bool(validation.get("passed", False)), "check_count": len(checks),
            "failures": failures, "validation_path": str(path / "validation.json")}


def _load_case(root, case, expected_mode):
    directory = root / case
    summary = _json(directory / "summary.json")
    if summary is None:
        raise FileNotFoundError(directory / "summary.json")
    settings = summary.get("settings", {})
    if settings.get("robust_mode") != expected_mode:
        raise ValueError(f"{directory}: expected explicit robust_mode={expected_mode!r}, "
                         f"got {settings.get('robust_mode')!r}")
    if float(settings.get("rho_hat", 0.)) <= 0:
        raise ValueError(f"{directory}: missing positive paper rho_hat bound")
    if expected_mode == "boundary_layer" and float(settings.get("robust_width", 0.)) <= 0:
        raise ValueError(f"{directory}: boundary-layer width must be positive")
    path = directory / "trajectory.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ("t", "error", "internal_force", "desired_internal_force")}
    t = arrays["t"]
    if t.ndim != 1 or len(t) < 3 or not np.all(np.diff(t) > 0):
        raise ValueError(f"{directory}: invalid recorded time grid")
    if t[0] > 2. + 1e-9 or t[-1] < 14. - 1e-9:
        raise ValueError(f"{directory}: complete [2,14] and [12,14] s records required")
    dt = float(np.median(np.diff(t)))
    if np.max(np.abs(np.diff(t) - dt)) > 1e-10:
        raise ValueError(f"{directory}: Welch requires uniform recorded control ticks")
    for key in ("error", "internal_force", "desired_internal_force"):
        shape = (len(t), 6) if key == "error" else (len(t), 2, 6)
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{directory}: invalid {key} shape or nonfinite samples")
    arrays["force_error"] = arrays["internal_force"] - arrays["desired_internal_force"]
    return {"directory": directory, "summary": summary, "settings": settings,
            "arrays": arrays, "dt": dt, "validation": _validation(directory),
            "trajectory_sha256": _sha256(path)}


def _power_metrics(values, dt):
    """Parseval-consistent, one-sided rectangular-window AC power fractions."""
    centered = values - np.mean(values, axis=0)
    count = len(values)
    frequency = np.fft.rfftfreq(count, dt)
    power = np.abs(np.fft.rfft(centered, axis=0) / count) ** 2
    power[1:] *= 2.
    if count % 2 == 0:
        power[-1] *= .5
    variance = np.mean(centered ** 2, axis=0)
    high_variance = np.sum(power[frequency >= 100.], axis=0)
    high_fraction = np.divide(high_variance, variance, out=np.zeros_like(variance), where=variance > 0)
    nyquist_variance = np.sum(power[frequency >= .4 / dt], axis=0)
    nyquist_fraction = np.divide(nyquist_variance, variance, out=np.zeros_like(variance), where=variance > 0)
    return np.sqrt(high_variance), high_fraction, nyquist_fraction


def _metrics(record):
    arrays, dt = record["arrays"], record["dt"]
    t = arrays["t"]
    final = (t >= 12. - 1e-9) & (t <= 14. + 1e-9)
    tracking = (t >= 2. - 1e-9) & (t <= 14. + 1e-9)
    force = arrays["force_error"][final]
    mean = np.mean(force, axis=0)
    ac = force - mean
    high_rms, high_fraction, nyquist_fraction = _power_metrics(force, dt)
    frequencies, psd = welch(ac, fs=1. / dt, nperseg=min(2048, len(force)), axis=0)
    peak_frequency = frequencies[np.argmax(psd[1:], axis=0) + 1]
    tracking_error = arrays["error"][tracking]
    result = {"robust_mode": record["settings"]["robust_mode"],
              "algorithm_variant": record["summary"].get("algorithm_variant"),
              "robust_width": record["settings"].get("robust_width"),
              "rho_hat": record["settings"]["rho_hat"],
              "sample_count": len(t), "control_dt_s": dt,
              "physics_dt_s": record["summary"]["physics_dt_s"],
              "force_window_s": [12., 14.], "force_window_sample_count": int(np.sum(final)),
              "force_mean_error": mean.tolist(),
              "force_rms_error": np.sqrt(np.mean(force ** 2, axis=0)).tolist(),
              "force_ac_rms_error": np.sqrt(np.mean(ac ** 2, axis=0)).tolist(),
              "force_peak_to_peak_error": np.ptp(force, axis=0).tolist(),
              "force_max_absolute_error": np.max(np.abs(force), axis=0).tolist(),
              "force_high_frequency_cutoff_Hz": 100.,
              "force_high_frequency_ac_rms": high_rms.tolist(),
              "force_high_frequency_ac_variance_fraction": high_fraction.tolist(),
              "force_top_20pct_nyquist_ac_variance_fraction": nyquist_fraction.tolist(),
              "force_welch_peak_frequency_Hz": peak_frequency.tolist(),
              "force_vector_rms_N": np.sqrt(np.mean(np.sum(force[:, :, :3] ** 2, axis=-1), axis=0)).tolist(),
              "generalized_moment_vector_rms_Nm": np.sqrt(np.mean(np.sum(force[:, :, 3:] ** 2, axis=-1), axis=0)).tolist(),
              "internal_force_zero_sum_max_abs": float(np.max(np.abs(np.sum(arrays["internal_force"], axis=1)))),
              "tracking_window_s": [2., 14.],
              "position_rmse_mm": float(1000. * np.sqrt(np.mean(np.sum(tracking_error[:, :3] ** 2, axis=1)))),
              "orientation_euler_rmse_deg": float(180. / np.pi * np.sqrt(np.mean(np.sum(tracking_error[:, 3:] ** 2, axis=1)))),
              "max_command_torque_limit_ratio": record["summary"].get("max_command_torque_limit_ratio"),
              "saturated_control_samples": record["summary"].get("saturated_control_samples"),
              "validation": record["validation"],
              "trajectory_path": str(record["directory"] / "trajectory.npz"),
              "trajectory_sha256": record["trajectory_sha256"]}
    return result


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": .5,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "path.simplify": False, "agg.path.chunksize": 10000,
                         "savefig.facecolor": "white"})


def _save(fig, output, stem):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def _label(record):
    if record["settings"]["robust_mode"] == "paper_sign":
        return "Paper sign"
    return f"Boundary layer ε={float(record['settings']['robust_width']):g}"


def _plot_waveforms(records, metrics, output):
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.), sharex=True)
    for row, (case, case_title) in enumerate(zip(CASES, TITLES)):
        for mode in ("paper_sign", "boundary_layer"):
            record = records[case][mode]
            t, error = record["arrays"]["t"], record["arrays"]["force_error"][:, 0, 1]
            window = (t >= 13.9 - 1e-9) & (t <= 14. + 1e-9)
            axes[row, 0].plot(t[window], error[window], color=COLORS[mode],
                              linewidth=.8, label=_label(record))
            if mode == "boundary_layer":
                axes[row, 1].plot(t[window], error[window], color=COLORS[mode], linewidth=.9)
                low, high = float(np.min(error[window])), float(np.max(error[window]))
                pad = max((high - low) * .15, 1e-6)
                axes[row, 1].set_ylim(low - pad, high + pad)
                axes[row, 1].ticklabel_format(useOffset=False, axis="y")
        axes[row, 0].set(title=f"{case_title}: same physical scale", ylabel="Left Y force error (N)")
        engineering = metrics[case]["boundary_layer"]
        rms, peak = engineering["force_rms_error"][0][1], engineering["force_peak_to_peak_error"][0][1]
        axes[row, 1].set(title=f"Boundary layer zoom | 12–14 s RMS={rms:.4g} N; p–p={peak:.4g} N",
                         ylabel="Left Y force error (N)")
        axes[row, 0].axhline(0., color="#777777", linewidth=.6)
        for ax in axes[row]:
            ax.set_xlim(13.9, 14.)
            ax.ticklabel_format(useOffset=False, axis="x")
            ax.set_xlabel("Time (s)")
        axes[row, 0].legend(loc="best")
    fig.suptitle("Raw internal-force error: final 0.1 s | paper sign vs boundary-layer engineering variant", y=.995)
    fig.text(.5, .009, "All recorded control ticks are connected; no low-pass filter or downsampling. Right-arm errors have opposite sign.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .026, 1, .97))
    _save(fig, output, "01_raw_Fy_error")


def _plot_spectra(records, output):
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.))
    for ax, case, case_title in zip(axes, CASES, TITLES):
        nyquist = []
        for mode in ("paper_sign", "boundary_layer"):
            record = records[case][mode]
            t, error = record["arrays"]["t"], record["arrays"]["force_error"][:, 0, 1]
            window = (t >= 12. - 1e-9) & (t <= 14. + 1e-9)
            centered = error[window] - np.mean(error[window])
            frequency, density = welch(centered, fs=1. / record["dt"], nperseg=min(2048, len(centered)))
            ax.semilogy(frequency[1:], np.maximum(density[1:], np.finfo(float).tiny),
                        color=COLORS[mode], linewidth=1., label=_label(record))
            nyquist.append(.5 / record["dt"])
        ax.axvline(100., color="#777777", linestyle="--", linewidth=.8, label="100 Hz metric cutoff")
        ax.set(xlim=(0, max(nyquist)), title=f"{case_title}: 12–14 s, left Y force error (mean removed)",
               xlabel="Frequency (Hz)", ylabel="PSD (N² / Hz)")
        ax.legend(loc="best", ncol=3)
    fig.suptitle("Internal-force error spectrum | paper sign vs boundary-layer engineering variant", y=.995)
    fig.text(.5, .009, "Welch: Hann window, up to 2048 samples, 50% overlap. Reported ≥100 Hz fractions use Parseval-consistent FFT power.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .026, 1, .97))
    _save(fig, output, "02_Fy_error_PSD")


def _report(records, metrics, output, paper_root, engineering_root, refinement=None):
    widths = sorted({float(records[case]["boundary_layer"]["settings"]["robust_width"]) for case in CASES})
    lines = ["# 降抖复核报告", "", "本报告从保存的原始轨迹重新计算；统计没有低通、降采样或删除尖峰。",
             f"论文模式目录：`{paper_root}`；工程模式目录：`{engineering_root}`。", "",
             "## 方法与理论范围", "",
             "`paper_sign` 使用论文的分量 sign 鲁棒项，保留正的 rho 上界。`boundary_layer` 将其中的 "
             "sign(w) 替换为 clip(w/ε,−1,1)，w=Y S；" + f"本次 ε={', '.join(f'{width:g}' for width in widths)}。"
             "ε 是 Y S 坐标中的边界层宽度，单位不能解释为 N。这是工程降抖变体，改变了原文不连续鲁棒项，原有精确到达证明不能直接沿用。",
             "", "论文原理论仅保证内力误差有界，并未保证其趋于零。此处数值改善属于保存采样轨迹的仿真证据。"
             "提取定义和约束隔离检查见 `../docs/internal_force_recheck.md`。", "",
             "力误差为 f_I−f_d；位置和姿态指标取闭区间 2–14 s，内力指标取闭区间 12–14 s。"
             "RMS 包含均值偏差；AC RMS 为去均值 RMS。≥100 Hz 占比指该频段占去均值误差总方差的比例，"
             "采用同一物理频率截止。后三个力分量是 Euler 广义力矩，单位 N m。", "",
             (f"工程变体名义工况左手 Y 平均误差仍为 {metrics['nominal']['boundary_layer']['force_mean_error'][0][1]:+.7f} N，"
              f"总 RMS {metrics['nominal']['boundary_layer']['force_rms_error'][0][1]:.7f} N，"
              f"AC RMS {metrics['nominal']['boundary_layer']['force_ac_rms_error'][0][1]:.8f} N。"
              "边界层大幅降低高频纹波，平均力静差仍然存在，不能把低 AC RMS 当作总误差趋零。"), "",
             "## 三种工况的 Y 向原始内力误差", "",
             "以下给出左手；右手均值取反，RMS、AC RMS、峰峰值及功率谱相同（两手零和残差另存 JSON）。", "",
             "| 工况 | 模式 | 均值 (N) | RMS (N) | AC RMS (N) | 峰峰 (N) | ≥100 Hz AC RMS (N) | ≥100 Hz 方差 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for case in CASES:
        for mode in ("paper_sign", "boundary_layer"):
            m = metrics[case][mode]
            values = [m[key][0][1] for key in ("force_mean_error", "force_rms_error", "force_ac_rms_error",
                                              "force_peak_to_peak_error", "force_high_frequency_ac_rms")]
            fraction = 100. * m["force_high_frequency_ac_variance_fraction"][0][1]
            lines.append(f"| {case} | {mode} | " + " | ".join(f"{value:.7g}" for value in values) + f" | {fraction:.5g}% |")
    lines += ["", "## 运动精度、三维内力及验证", "",
              "| 工况 | 模式 | 位置向量 RMSE (mm) | Euler 角向量 RMSE (deg) | 三维力误差 RMS (N) | 三维广义力矩误差 RMS (N m) | 最大扭矩/限值 | 独立验证 |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for case in CASES:
        for mode in ("paper_sign", "boundary_layer"):
            m = metrics[case][mode]
            ratio = m["max_command_torque_limit_ratio"]
            ratio_text = f"{ratio:.6g}" if ratio is not None else "未记录"
            v = m["validation"]
            lines.append(f"| {case} | {mode} | {m['position_rmse_mm']:.7g} | {m['orientation_euler_rmse_deg']:.7g} | "
                         f"{m['force_vector_rms_N'][0]:.7g} | {m['generalized_moment_vector_rms_Nm'][0]:.7g} | "
                         f"{ratio_text} | {v['status']} ({v['check_count']}) |")
    lines += ["", "## 参数一致性与降幅", ""]
    for case in CASES:
        paper, engineering = metrics[case]["paper_sign"], metrics[case]["boundary_layer"]
        rms_old, rms_new = paper["force_rms_error"][0][1], engineering["force_rms_error"][0][1]
        ac_old, ac_new = paper["force_ac_rms_error"][0][1], engineering["force_ac_rms_error"][0][1]
        rms_change = 100. * (1. - rms_new / rms_old) if rms_old > 0 else 0.
        ac_change = 100. * (1. - ac_new / ac_old) if ac_old > 0 else 0.
        a, b = records[case]["paper_sign"]["settings"], records[case]["boundary_layer"]["settings"]
        different = {key: [a.get(key), b.get(key)] for key in sorted(set(a) | set(b))
                     if key not in ("robust_mode", "robust_width") and a.get(key) != b.get(key)}
        time_match = abs(paper["control_dt_s"] - engineering["control_dt_s"]) < 1e-12 and abs(paper["physics_dt_s"] - engineering["physics_dt_s"]) < 1e-12
        equality = "其余 Settings 一致" if not different else "其余 Settings 存在差异：" + json.dumps(different, ensure_ascii=False)
        lines.append(f"- {case}：Y 总 RMS 降幅 {rms_change:.5g}%，AC RMS 降幅 {ac_change:.5g}%；{equality}；"
                     f"控制/物理步长{'一致' if time_match else '不同'}。")
    lines += ["", "## 完整六分量数据", "",
              "JSON 的各 2×6 数组按 [左手,右手]、[Fx,Fy,Fz,m_roll,m_pitch,m_yaw] 排列。以下为左手，"
              "右手完整数值也保存在 JSON；不能把 N 和 N m 混合成一个有单位的误差范数。", ""]
    for case in CASES:
        lines += [f"### {case}", "", "| 分量 | 模式 | 单位 | 均值 | RMS | AC RMS | 峰峰 | ≥100 Hz 方差 |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for j, (component, unit) in enumerate(zip(COMPONENTS, UNITS)):
            for mode in ("paper_sign", "boundary_layer"):
                m = metrics[case][mode]
                values = [m[key][0][j] for key in ("force_mean_error", "force_rms_error", "force_ac_rms_error", "force_peak_to_peak_error")]
                fraction = 100. * m["force_high_frequency_ac_variance_fraction"][0][j]
                lines.append(f"| {component} | {mode} | {unit} | " + " | ".join(f"{value:.7g}" for value in values) + f" | {fraction:.5g}% |")
        lines.append("")
    if refinement is not None:
        lines += ["## 工程变体的名义工况半步长复核", "",
                  "下列比较均为 boundary_layer；用相同的 12–14 s 力统计窗口、2–14 s 运动窗口。", "",
                  "| 控制/物理步长 (ms) | Y 均值 (N) | Y RMS (N) | Y AC RMS (N) | Y 峰峰 (N) | 位置 RMSE (mm) | Euler RMSE (deg) | 验证 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for item in (metrics["nominal"]["boundary_layer"], refinement):
            v = item["validation"]
            lines.append(f"| {1000*item['control_dt_s']:.4g} / {1000*item['physics_dt_s']:.4g} | "
                         f"{item['force_mean_error'][0][1]:.7g} | {item['force_rms_error'][0][1]:.7g} | "
                         f"{item['force_ac_rms_error'][0][1]:.7g} | {item['force_peak_to_peak_error'][0][1]:.7g} | "
                         f"{item['position_rmse_mm']:.7g} | {item['orientation_euler_rmse_deg']:.7g} | "
                         f"{v['status']} ({v['check_count']}) |")
        lines.append("")
    lines += ["## 图与可追溯性", "",
              "- `01_raw_Fy_error.png/.pdf`：13.9–14 s 原始控制采样误差；右列单独放大工程变体，未平滑。",
              "- `02_Fy_error_PSD.png/.pdf`：12–14 s 去均值误差的 Welch 功率谱。",
              "- `metrics.json`：六分量/双手统计、原轨迹绝对路径及 SHA256、实际步长、设置和验证状态。", "",
              "约束力取每次新保持扭矩施加后的瞬时右极限。此报告不将控制时刻采样值当作区间平均力，"
              "不宣称精确 Filippov 到达或实体机械臂已经验证。验证显示 MISSING 时表示独立验证文件尚未生成，不能解释为通过。", ""]
    lines += ["工程变体末段 AC RMS 包含缓慢漂移。未加窗 FFT 的高频占比会包含有限窗口边界的谱泄漏，"
              "不能把该份额全部解释成真实高频抖振；图中的 Hann-Welch PSD 与表格的 FFT 功率采用不同的窗定义。", ""]
    lines += ["模式判定以 summary 的 `settings.robust_mode` 和 `algorithm_variant` 为准；"
              "遗留的通用 `claim_limit` 字符串不代表 boundary_layer 仍使用原 sign 控制。", ""]
    (output / "降抖复核报告.md").write_text("\n".join(lines), encoding="utf-8")


def compare_results(paper_results, engineering_results, output):
    paper_root, engineering_root, output = map(lambda path: Path(path).resolve(),
                                                (paper_results, engineering_results, output))
    records, metrics = {}, {}
    for case in CASES:
        records[case] = {"paper_sign": _load_case(paper_root, case, "paper_sign"),
                         "boundary_layer": _load_case(engineering_root, case, "boundary_layer")}
        metrics[case] = {mode: _metrics(record) for mode, record in records[case].items()}
    refinement = None
    refinement_root = engineering_root / "step_refinement"
    if (refinement_root / "nominal" / "summary.json").exists() and (refinement_root / "nominal" / "validation.json").exists():
        refinement_record = _load_case(refinement_root, "nominal", "boundary_layer")
        refinement = _metrics(refinement_record)
        refinement["settings"] = refinement_record["settings"]
    output.mkdir(parents=True, exist_ok=True)
    payload = {"paper_results": str(paper_root), "engineering_results": str(engineering_root),
               "component_order": list(COMPONENTS), "component_units": list(UNITS),
               "hand_order": ["left", "right"], "cases": metrics,
               "settings": {case: {mode: rec["settings"] for mode, rec in records[case].items()} for case in CASES},
               "aggregate_validation": {"paper_sign": _json(paper_root / "validation_summary.json"),
                                        "boundary_layer": _json(engineering_root / "validation_summary.json")},
               "engineering_step_refinement": refinement,
               "interpretation": "Raw sampled torque-driven MuJoCo force; boundary_layer changes the original paper sign law"}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _style()
    _plot_waveforms(records, metrics, output)
    _plot_spectra(records, output)
    _report(records, metrics, output, paper_root, engineering_root, refinement)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-results", type=Path, default=HERE / "results")
    parser.add_argument("--engineering-results", type=Path, default=HERE / "results_low_chatter")
    parser.add_argument("--output", type=Path, default=HERE / "recheck")
    args = parser.parse_args()
    payload = compare_results(args.paper_results, args.engineering_results, args.output)
    print(f"Comparison report, metrics.json, and two PNG/PDF pairs: {args.output.resolve()}")
    for case, modes in payload["cases"].items():
        print(f"{case}: Y force RMS {modes['paper_sign']['force_rms_error'][0][1]:.6g} -> "
              f"{modes['boundary_layer']['force_rms_error'][0][1]:.6g} N", flush=True)


if __name__ == "__main__":
    main()
