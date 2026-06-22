from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("output/turning_point_early_warning_rise/turning_point_metric_table.csv")
OUTPUT_DIR = Path("output/turning_point_early_warning_rise")

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


def safe_dataset(dataset: str) -> str:
    return dataset.replace("GÉANT", r"G\'EANT")


def build_full_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        model_df = df[df["Model"] == model]
        if model_df.empty:
            continue
        row = {"Method": model}
        fars = []
        lead_sum = 0.0
        for dataset in ["Abilene", "GÉANT"]:
            ds = model_df[model_df["Dataset"] == dataset]
            if ds.empty:
                row[f"{dataset} Mean Lead"] = "-"
                row[f"{dataset} FAR (%)"] = "-"
                continue
            item = ds.iloc[0]
            mean_lead = item["MeanLead"]
            far = item["FARAlarmPct"]
            row[f"{dataset} Mean Lead"] = "0.0" if pd.isna(mean_lead) else f"{float(mean_lead):.1f}"
            row[f"{dataset} FAR (%)"] = f"{float(far):.2f}"
            if not pd.isna(mean_lead):
                lead_sum += float(mean_lead)
            fars.append(float(far))
        row["Avg. Mean Lead"] = f"{lead_sum / 2.0:.1f}"
        row["Avg. FAR (%)"] = f"{sum(fars) / len(fars):.2f}" if fars else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def write_md(table: pd.DataFrame, path: Path) -> None:
    headers = list(table.columns)
    values = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [
        max(len(header), *(len(row[i]) for row in values)) if values else len(header)
        for i, header in enumerate(headers)
    ]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    lines = [
        fmt(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt(row) for row in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(table: pd.DataFrame, path: Path) -> None:
    metric_columns = [
        ("Abilene Mean Lead", "higher"),
        ("Abilene FAR (%)", "lower"),
        ("GÉANT Mean Lead", "higher"),
        ("GÉANT FAR (%)", "lower"),
        ("Avg. FAR (%)", "lower"),
    ]

    ranks: dict[str, dict[str, set[int]]] = {}
    for column, direction in metric_columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        values = sorted(numeric.dropna().unique(), reverse=(direction == "higher"))
        best = values[0] if values else None
        second = values[1] if len(values) > 1 else None
        ranks[column] = {
            "best": set(numeric[numeric == best].index.tolist()) if best is not None else set(),
            "second": set(numeric[numeric == second].index.tolist()) if second is not None else set(),
        }

    def fmt_cell(row_idx: int, column: str, value: object) -> str:
        text = str(value)
        if row_idx in ranks.get(column, {}).get("best", set()):
            return rf"\textbf{{{text}}}"
        if row_idx in ranks.get(column, {}).get("second", set()):
            return rf"\underline{{{text}}}"
        return text

    lines = [
        r"% Requires: \usepackage{graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.8pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        (
            r"\textbf{Method}"
            r" & \multicolumn{2}{c}{\textbf{Abilene}}"
            r" & \multicolumn{2}{c}{\textbf{G\'EANT}}"
            r" & \textbf{Avg. FAR}"
            r"\tabularnewline"
        ),
        (
            r" & \textbf{Mean Lead $\uparrow$} & \textbf{FAR (\%) $\downarrow$}"
            r" & \textbf{Mean Lead $\uparrow$} & \textbf{FAR (\%) $\downarrow$}"
            r" & \textbf{(\%) $\downarrow$}"
            r"\tabularnewline"
        ),
        r"\hline",
    ]

    for row_idx, row in table.iterrows():
        values = [
            str(row["Method"]).replace("_", r"\_"),
            fmt_cell(row_idx, "Abilene Mean Lead", row["Abilene Mean Lead"]),
            fmt_cell(row_idx, "Abilene FAR (%)", row["Abilene FAR (%)"]),
            fmt_cell(row_idx, "GÉANT Mean Lead", row["GÉANT Mean Lead"]),
            fmt_cell(row_idx, "GÉANT FAR (%)", row["GÉANT FAR (%)"]),
            fmt_cell(row_idx, "Avg. FAR (%)", row["Avg. FAR (%)"]),
        ]
        lines.append(" & ".join(values) + r"\tabularnewline")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{Compact neutral turning-point early-warning comparison without averaging baselines. "
                r"Mean Lead measures the average number of time slots by which successful alarms are triggered "
                r"before the true threshold crossing. FAR is the false alarm rate among alarm windows. "
                r"Higher Mean Lead and lower FAR are better. The best result in each metric is highlighted "
                r"in bold and the second-best result is underlined.}"
            ),
            r"\label{tab:turning_point_compact_full_baselines}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tex_with_avg_lead(table: pd.DataFrame, path: Path) -> None:
    metric_columns = [
        ("Abilene Mean Lead", "higher"),
        ("Abilene FAR (%)", "lower"),
        ("GÉANT Mean Lead", "higher"),
        ("GÉANT FAR (%)", "lower"),
        ("Avg. Mean Lead", "higher"),
        ("Avg. FAR (%)", "lower"),
    ]

    ranks: dict[str, dict[str, set[int]]] = {}
    for column, direction in metric_columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        values = sorted(numeric.dropna().unique(), reverse=(direction == "higher"))
        best = values[0] if values else None
        second = values[1] if len(values) > 1 else None
        ranks[column] = {
            "best": set(numeric[numeric == best].index.tolist()) if best is not None else set(),
            "second": set(numeric[numeric == second].index.tolist()) if second is not None else set(),
        }

    def fmt_cell(row_idx: int, column: str, value: object) -> str:
        text = str(value)
        if row_idx in ranks.get(column, {}).get("best", set()):
            return rf"\textbf{{{text}}}"
        if row_idx in ranks.get(column, {}).get("second", set()):
            return rf"\underline{{{text}}}"
        return text

    lines = [
        r"% Requires: \usepackage{graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        (
            r"\textbf{Method}"
            r" & \multicolumn{2}{c}{\textbf{Abilene}}"
            r" & \multicolumn{2}{c}{\textbf{G\'EANT}}"
            r" & \textbf{Avg. Lead}"
            r" & \textbf{Avg. FAR}"
            r"\tabularnewline"
        ),
        (
            r" & \textbf{Mean Lead $\uparrow$} & \textbf{FAR (\%) $\downarrow$}"
            r" & \textbf{Mean Lead $\uparrow$} & \textbf{FAR (\%) $\downarrow$}"
            r" & \textbf{$\uparrow$}"
            r" & \textbf{(\%) $\downarrow$}"
            r"\tabularnewline"
        ),
        r"\hline",
    ]

    for row_idx, row in table.iterrows():
        values = [
            str(row["Method"]).replace("_", r"\_"),
            fmt_cell(row_idx, "Abilene Mean Lead", row["Abilene Mean Lead"]),
            fmt_cell(row_idx, "Abilene FAR (%)", row["Abilene FAR (%)"]),
            fmt_cell(row_idx, "GÉANT Mean Lead", row["GÉANT Mean Lead"]),
            fmt_cell(row_idx, "GÉANT FAR (%)", row["GÉANT FAR (%)"]),
            fmt_cell(row_idx, "Avg. Mean Lead", row["Avg. Mean Lead"]),
            fmt_cell(row_idx, "Avg. FAR (%)", row["Avg. FAR (%)"]),
        ]
        lines.append(" & ".join(values) + r"\tabularnewline")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{Compact neutral turning-point early-warning comparison without averaging baselines. "
                r"Mean Lead measures the average number of time slots by which successful alarms are triggered "
                r"before the true threshold crossing. Avg. Lead is averaged over the two datasets, with missing "
                r"successful early alarms counted as zero lead. FAR is the false alarm rate among alarm windows. "
                r"The best result in each metric is highlighted in bold and the second-best result is underlined.}"
            ),
            r"\label{tab:turning_point_compact_full_baselines_avg_lead}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    table = build_full_baseline_table(df)
    prefix_avg_lead = OUTPUT_DIR / "turning_point_compact_full_baselines_with_avg_lead_table"
    table.to_csv(prefix_avg_lead.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    write_md(table, prefix_avg_lead.with_suffix(".md"))
    write_tex_with_avg_lead(table, prefix_avg_lead.with_suffix(".tex"))
    print(f"Saved compact full-baseline table with avg lead to {prefix_avg_lead}.[csv|md|tex]")


if __name__ == "__main__":
    main()
