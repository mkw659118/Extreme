from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "DATP-Net",
    "PMDformer",
    "HMformer",
    "FeTS",
    "TimesNet",
    "iTransformer",
    "PatchTST",
    "WPMixer",
    "P-sLSTM",
    "xLSTMTime",
    "xLSTM-Mixer",
    "FEDformer",
]


def format_float(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def build_table(input_dir: Path) -> pd.DataFrame:
    alarm = pd.read_csv(input_dir / "event_alarm_summary.csv")
    false_alarm = pd.read_csv(input_dir / "false_alarm_summary.csv")

    df = alarm.merge(
        false_alarm[
            [
                "dataset",
                "model",
                "label",
                "alarm_windows",
                "false_alarm_windows",
                "false_alarm_rate_among_alarms",
                "false_alarm_rate_among_windows",
            ]
        ],
        on=["dataset", "model", "label"],
        how="left",
    )

    df["early_alarm_rate"] = df["early_or_on_time_count"] / df["num_cases"]
    df["miss_rate"] = df["missed_count"] / df["num_cases"]
    df["delay_rate"] = df["delayed_count"] / df["num_cases"]
    df["false_alarm_rate_percent"] = df["false_alarm_rate_among_alarms"] * 100.0
    df["window_false_alarm_rate_percent"] = df["false_alarm_rate_among_windows"] * 100.0

    df["model_order"] = df["label"].map({label: idx for idx, label in enumerate(MODEL_ORDER)}).fillna(999)
    df = df.sort_values(["dataset", "model_order", "label"]).reset_index(drop=True)

    return df[
        [
            "dataset",
            "label",
            "num_cases",
            "early_or_on_time_count",
            "early_alarm_rate",
            "delayed_count",
            "delay_rate",
            "missed_count",
            "miss_rate",
            "mean_lead_time_slots",
            "median_lead_time_slots",
            "alarm_windows",
            "false_alarm_windows",
            "false_alarm_rate_percent",
            "window_false_alarm_rate_percent",
        ]
    ].rename(
        columns={
            "dataset": "Dataset",
            "label": "Model",
            "num_cases": "Cases",
            "early_or_on_time_count": "Early",
            "early_alarm_rate": "EarlyRate",
            "delayed_count": "Delayed",
            "delay_rate": "DelayedRate",
            "missed_count": "Missed",
            "miss_rate": "MissRate",
            "mean_lead_time_slots": "MeanLead",
            "median_lead_time_slots": "MedianLead",
            "alarm_windows": "AlarmWindows",
            "false_alarm_windows": "FalseAlarmWindows",
            "false_alarm_rate_percent": "FARAlarmPct",
            "window_false_alarm_rate_percent": "FARWindowPct",
        }
    )


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Dataset"] = df["Dataset"]
    out["Model"] = df["Model"]
    out["Early"] = df["Early"].astype(int).astype(str) + "/" + df["Cases"].astype(int).astype(str)
    out["Miss"] = df["Missed"].astype(int)
    out["Delay"] = df["Delayed"].astype(int)
    out["Mean Lead"] = df["MeanLead"].map(lambda x: format_float(x, 1))
    out["Median Lead"] = df["MedianLead"].map(lambda x: format_float(x, 1))
    out["FAR (%)"] = df["FARAlarmPct"].map(lambda x: format_float(x, 2))
    return out


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[col_idx]) for row in rows)) if rows else len(str(header))
        for col_idx, header in enumerate(headers)
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    lines = [
        fmt_row([str(header) for header in headers]),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(df: pd.DataFrame, path: Path) -> None:
    def esc(value: object) -> str:
        text = str(value)
        text = text.replace("GÉANT", r"G\'EANT")
        text = text.replace("_", r"\_")
        text = text.replace("%", r"\%")
        return text

    lines = [
        r"% Requires: \usepackage{graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\hline",
        (
            r"\textbf{Dataset} & \textbf{Model} & \textbf{Early} & "
            r"\textbf{Miss} & \textbf{Delay} & \textbf{Mean Lead} & "
            r"\textbf{Median Lead} & \textbf{FAR (\%)}"
            r"\tabularnewline"
        ),
        r"\hline",
    ]

    for _, row in df.iterrows():
        values = [
            esc(row["Dataset"]),
            esc(row["Model"]),
            esc(row["Early"]),
            esc(row["Miss"]),
            esc(row["Delay"]),
            esc(row["Mean Lead"]),
            esc(row["Median Lead"]),
            esc(row["FAR (%)"]),
        ]
        lines.append(" & ".join(values) + r"\tabularnewline")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{Neutral turning-point early-warning results. "
                r"Early indicates alarms triggered before or at the true threshold crossing. "
                r"Mean Lead and Median Lead are measured in time slots. "
                r"FAR is the false alarm rate among alarm windows.}"
            ),
            r"\label{tab:turning_point_early_warning}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build turning-point early-warning metric table.")
    parser.add_argument("--input-dir", type=Path, default=Path("output/turning_point_early_warning_rise"))
    parser.add_argument("--output-prefix", type=Path, default=Path("output/turning_point_early_warning_rise/turning_point_metric_table"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    raw = build_table(args.input_dir)
    pretty = display_table(raw)

    raw.to_csv(args.output_prefix.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    pretty.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_display.csv"), index=False, encoding="utf-8-sig")
    write_markdown(pretty, args.output_prefix.with_suffix(".md"))
    write_latex(pretty, args.output_prefix.with_suffix(".tex"))

    print(f"Saved metric table to {args.output_prefix}.[csv|md|tex]")


if __name__ == "__main__":
    main()
