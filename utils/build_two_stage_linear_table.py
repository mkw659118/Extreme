"""Build paper-style tables from two-stage linear baseline result pickles."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path


METRICS = ("NMAE", "NRMSE", "COS")
DATASETS = ("Abilene", "Geant")
MODEL_ORDER = (
    "PMDformer",
    "HMformer",
    "FeTS",
    "timesnet",
    "iTransformer",
    "FEDformer",
    "PatchTST",
    "WPMixer",
    "P_sLSTM",
    "xLSTMTime",
    "xlstm_mixer",
)
MODEL_LABELS = {
    "timesnet": "TimesNet",
    "xlstm_mixer": "xLSTM-Mixer",
    "P_sLSTM": "P-sLSTM",
}
SCENARIOS = (
    ("random_5", "Random Missing Rate = 5%", "random_point", 0.05),
    (
        "structured_20",
        "Structured Time-Block Missing Rate = 20%",
        "time_block",
        0.20,
    ),
)


@dataclass
class Result:
    path: Path
    config: dict
    metrics: dict[str, tuple[float, float]]


def _normal_pattern(value) -> str:
    value = str(value or "random_point").strip().lower().replace("-", "_")
    if value in {"random", "point"}:
        return "random_point"
    if value in {"block", "structured_block"}:
        return "time_block"
    return value


def _load_result(path: Path) -> Result | None:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        print(f"[Table] skip unreadable result {path}: {exc}")
        return None

    config = payload.get("config", {})
    if not isinstance(config, dict):
        return None

    metrics = {}
    for metric in METRICS:
        mean = payload.get(f"{metric}_mean")
        std = payload.get(f"{metric}_std")
        if mean is None:
            values = payload.get(metric)
            if values is not None:
                try:
                    values = [float(value) for value in values]
                    mean = sum(values) / len(values)
                    variance = sum((value - mean) ** 2 for value in values) / len(values)
                    std = math.sqrt(variance)
                except (TypeError, ValueError, ZeroDivisionError):
                    mean = None
        if mean is not None:
            metrics[metric] = (float(mean), float(std or 0.0))
    return Result(path=path, config=config, metrics=metrics)


def _matches(result: Result, dataset: str, model: str, pattern: str, rate: float, pred_len: int, d_model: int) -> bool:
    config = result.config
    return (
        str(config.get("input_imputation", "")).lower() == "linear"
        and bool(config.get("two_stage_forecasting", False))
        and str(config.get("dataset", "")).lower() == dataset.lower()
        and str(config.get("model", "")) == model
        and _normal_pattern(config.get("artificial_missing_pattern")) == pattern
        and abs(float(config.get("artificial_missing_rate", -1.0)) - rate) < 1e-9
        and int(config.get("pred_len", -1)) == pred_len
        and int(config.get("d_model", -1)) == d_model
    )


def _format_value(
    value: tuple[float, float] | None,
    latex: bool = False,
    style: str | None = None,
) -> str:
    if value is None:
        return "--"
    mean, std = value
    if latex:
        text = f"{mean:.4f} $\\pm$ {std:.4f}"
        if style == "best":
            return f"\\textbf{{{text}}}"
        if style == "second":
            return f"\\underline{{{text}}}"
        return text
    text = f"{mean:.4f} ± {std:.4f}"
    if style == "best":
        return f"**{text}**"
    if style == "second":
        return f"<u>{text}</u>"
    return text


def _build_rank_styles(selected):
    """Mark the best and second-best mean for every table metric column."""

    styles = {}
    for scenario_key, _, _, _ in SCENARIOS:
        for dataset in DATASETS:
            for metric in METRICS:
                ranked = []
                for model in MODEL_ORDER:
                    result = selected.get((scenario_key, model, dataset))
                    if result and metric in result.metrics:
                        ranked.append((model, result.metrics[metric][0]))
                ranked.sort(key=lambda item: item[1], reverse=(metric == "COS"))
                if ranked:
                    styles[(scenario_key, ranked[0][0], dataset, metric)] = "best"
                if len(ranked) > 1:
                    styles[(scenario_key, ranked[1][0], dataset, metric)] = "second"
    return styles


def _collect(result_dir: Path, pred_len: int, d_model: int):
    loaded = [
        result
        for path in result_dir.glob("*.pkl")
        if (result := _load_result(path)) is not None
    ]
    selected = {}
    missing = []

    for scenario_key, _, pattern, rate in SCENARIOS:
        for model in MODEL_ORDER:
            for dataset in DATASETS:
                candidates = [
                    result
                    for result in loaded
                    if _matches(
                        result,
                        dataset,
                        model,
                        pattern,
                        rate,
                        pred_len,
                        d_model,
                    )
                ]
                if candidates:
                    selected[(scenario_key, model, dataset)] = max(
                        candidates,
                        key=lambda item: item.path.stat().st_mtime,
                    )
                else:
                    missing.append((scenario_key, model, dataset))
    return selected, missing


def _write_csv(path: Path, selected):
    header = ["scenario", "model"]
    for dataset in DATASETS:
        for metric in METRICS:
            header.extend([f"{dataset}_{metric}_mean", f"{dataset}_{metric}_std"])

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for scenario_key, scenario_label, _, _ in SCENARIOS:
            for model in MODEL_ORDER:
                row = [scenario_label, MODEL_LABELS.get(model, model)]
                for dataset in DATASETS:
                    result = selected.get((scenario_key, model, dataset))
                    for metric in METRICS:
                        value = result.metrics.get(metric) if result else None
                        row.extend(value if value else ("", ""))
                writer.writerow(row)


def _write_markdown(path: Path, selected, pred_len: int, d_model: int):
    styles = _build_rank_styles(selected)
    lines = [
        "# Two-stage linear-interpolation forecasting results",
        "",
        f"Prediction length: {pred_len}; d_model: {d_model}. Values are mean ± standard deviation.",
        "",
    ]
    header = (
        "| Model | Abilene NMAE↓ | Abilene NRMSE↓ | Abilene COS↑ | "
        "GÉANT NMAE↓ | GÉANT NRMSE↓ | GÉANT COS↑ |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|"

    for scenario_key, scenario_label, _, _ in SCENARIOS:
        lines.extend([f"## {scenario_label}", "", header, separator])
        for model in MODEL_ORDER:
            cells = [f"Linear + {MODEL_LABELS.get(model, model)}"]
            for dataset in DATASETS:
                result = selected.get((scenario_key, model, dataset))
                for metric in METRICS:
                    cells.append(
                        _format_value(
                            result.metrics.get(metric) if result else None,
                            style=styles.get((scenario_key, model, dataset, metric)),
                        )
                    )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex(path: Path, selected, pred_len: int, d_model: int):
    styles = _build_rank_styles(selected)
    lines = [
        "% Auto-generated by utils/build_two_stage_linear_table.py",
        f"% pred_len={pred_len}, d_model={d_model}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Forecasting performance on Abilene and G\\'EANT under incomplete historical observations. All baselines use linear interpolation before forecasting.}",
        "\\begin{tabular}{lccc|ccc}",
        "\\toprule",
        "& \\multicolumn{3}{c|}{Abilene} & \\multicolumn{3}{c}{G\\'EANT} \\\\",
        "Model & NMAE$\\downarrow$ & NRMSE$\\downarrow$ & COS$\\uparrow$ & NMAE$\\downarrow$ & NRMSE$\\downarrow$ & COS$\\uparrow$ \\\\",
        "\\midrule",
    ]
    for scenario_index, (scenario_key, scenario_label, _, _) in enumerate(SCENARIOS):
        if scenario_index:
            lines.append("\\midrule")
        latex_scenario_label = scenario_label.replace("%", "\\%")
        lines.append(f"\\multicolumn{{7}}{{c}}{{\\textit{{{latex_scenario_label}}}}} \\\\")
        for model in MODEL_ORDER:
            cells = [f"Linear + {MODEL_LABELS.get(model, model)}"]
            for dataset in DATASETS:
                result = selected.get((scenario_key, model, dataset))
                for metric in METRICS:
                    cells.append(
                        _format_value(
                            result.metrics.get(metric) if result else None,
                            latex=True,
                            style=styles.get((scenario_key, model, dataset, metric)),
                        )
                    )
            lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/two_stage_linear"))
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    selected, missing = _collect(args.result_dir, args.pred_len, args.d_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"table_iii_two_stage_linear_PL{args.pred_len}_DM{args.d_model}"
    csv_path = args.output_dir / f"{stem}.csv"
    md_path = args.output_dir / f"{stem}.md"
    tex_path = args.output_dir / f"{stem}.tex"
    _write_csv(csv_path, selected)
    _write_markdown(md_path, selected, args.pred_len, args.d_model)
    _write_latex(tex_path, selected, args.pred_len, args.d_model)

    print(f"[Table] CSV: {csv_path}")
    print(f"[Table] Markdown: {md_path}")
    print(f"[Table] LaTeX: {tex_path}")
    if missing:
        print(f"[Table] missing configurations: {len(missing)}")
        for scenario, model, dataset in missing[:20]:
            print(f"  - {scenario}: {dataset}/{model}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
