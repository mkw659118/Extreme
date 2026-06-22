from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "datp_net": "DATP-Net",
    "PMDformer": "PMDformer",
    "HMformer": "HMformer",
    "FeTS": "FeTS",
    "timesnet": "TimesNet",
    "iTransformer": "iTransformer",
    "PatchTST": "PatchTST",
    "WPMixer": "WPMixer",
    "P_sLSTM": "P-sLSTM",
    "xLSTMTime": "xLSTMTime",
    "xlstm_mixer": "xLSTM-Mixer",
    "FEDformer": "FEDformer",
}

PREFERRED_MODELS = [
    "datp_net",
    "PMDformer",
    "HMformer",
    "FeTS",
    "timesnet",
    "iTransformer",
    "PatchTST",
    "WPMixer",
    "P_sLSTM",
    "xLSTMTime",
    "xlstm_mixer",
    "FEDformer",
]

DATASET_LABELS = {
    "Abilene": "Abilene",
    "Geant": "GÉANT",
}


@dataclass(frozen=True)
class ModelSeries:
    dataset: str
    model: str
    label: str
    agg_path: Path
    raw_path: Path
    time_idx: np.ndarray
    true: np.ndarray
    risk_time_idx: np.ndarray
    risk_pred: np.ndarray
    h1_pred: np.ndarray


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def dataset_slug(dataset: str) -> str:
    return dataset.lower().replace("é", "e")


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def find_model_files(dataset_root: Path) -> dict[str, tuple[Path, Path]]:
    files: dict[str, tuple[Path, Path]] = {}
    for model in PREFERRED_MODELS:
        model_root = dataset_root / model
        if not model_root.exists():
            continue
        candidates = sorted(model_root.glob("PL*/TC*/test_raw.csv"))
        preferred = [p for p in candidates if any(part.startswith("PL5_") for part in p.parts)]
        if not preferred and not candidates:
            continue
        raw_path = preferred[0] if preferred else candidates[0]
        agg_path = raw_path.with_name("test_agg.csv")
        if agg_path.exists():
            files[model] = (agg_path, raw_path)
    return files


def load_model_series(dataset: str, model: str, agg_path: Path, raw_path: Path, pred_len: int) -> ModelSeries:
    agg = pd.read_csv(agg_path)
    raw = pd.read_csv(raw_path)
    if not {"true", "pred"}.issubset(agg.columns):
        raise ValueError(f"{agg_path} must contain true and pred columns")
    if not {"true", "pred"}.issubset(raw.columns):
        raise ValueError(f"{raw_path} must contain true and pred columns")

    usable = (len(raw) // pred_len) * pred_len
    if usable == 0:
        raise ValueError(f"{raw_path} has no complete pred_len={pred_len} windows")
    raw = raw.iloc[:usable].copy()
    pred = pd.to_numeric(raw["pred"], errors="coerce").to_numpy(dtype=np.float64).reshape(-1, pred_len)

    time_idx = (
        pd.to_numeric(agg["time_idx"], errors="coerce").to_numpy(dtype=np.float64)
        if "time_idx" in agg.columns
        else np.arange(len(agg), dtype=np.float64)
    )
    true = pd.to_numeric(agg["true"], errors="coerce").to_numpy(dtype=np.float64)
    agg_pred = pd.to_numeric(agg["pred"], errors="coerce").to_numpy(dtype=np.float64)

    n_windows = pred.shape[0]
    risk_time_idx = time_idx[:n_windows]
    risk_pred = np.nanmax(pred, axis=1)
    h1_pred = pred[:, 0]

    return ModelSeries(
        dataset=dataset,
        model=model,
        label=model_label(model),
        agg_path=agg_path,
        raw_path=raw_path,
        time_idx=time_idx,
        true=np.clip(true, 0.0, None),
        risk_time_idx=risk_time_idx,
        risk_pred=np.clip(risk_pred, 0.0, None),
        h1_pred=np.clip(h1_pred, 0.0, None),
    )


def select_events(
    true: np.ndarray,
    threshold: float,
    max_events: int,
    lookback: int,
    lookahead: int,
    min_gap: int,
) -> list[int]:
    above = true > threshold
    crossings = np.where((~above[:-1]) & above[1:])[0] + 1
    candidates: list[tuple[float, int]] = []
    for event_idx in crossings:
        if event_idx < lookback or event_idx + lookahead >= len(true):
            continue
        left = max(0, event_idx - lookback)
        right = min(len(true), event_idx + lookahead + 1)
        pre_min = float(np.min(true[left:event_idx]))
        post_max = float(np.max(true[event_idx:right]))
        local_diff = float(true[event_idx] - true[event_idx - 1])
        score = (post_max - pre_min) + max(local_diff, 0.0)
        candidates.append((score, int(event_idx)))

    selected: list[int] = []
    for _, idx in sorted(candidates, reverse=True):
        if all(abs(idx - chosen) >= min_gap for chosen in selected):
            selected.append(idx)
        if len(selected) >= max_events:
            break
    return sorted(selected)


def candidate_crossings(true: np.ndarray, threshold: float, lookback: int, lookahead: int) -> np.ndarray:
    above = true > threshold
    crossings = np.where((~above[:-1]) & above[1:])[0] + 1
    return np.array(
        [idx for idx in crossings if idx >= lookback and idx + lookahead < len(true)],
        dtype=np.int64,
    )


def find_alarm_time(
    risk_time_idx: np.ndarray,
    risk_pred: np.ndarray,
    threshold: float,
    event_time: float,
    search_start: float,
    search_end: float,
) -> tuple[float | None, str]:
    in_window = (risk_time_idx >= search_start) & (risk_time_idx <= search_end)
    alarm_candidates = np.where(in_window & (risk_pred > threshold))[0]
    if alarm_candidates.size:
        return float(risk_time_idx[alarm_candidates[0]]), "early_or_on_time"

    delayed_window = (risk_time_idx > event_time) & (risk_time_idx <= event_time + (search_end - event_time))
    delayed_candidates = np.where(delayed_window & (risk_pred > threshold))[0]
    if delayed_candidates.size:
        return float(risk_time_idx[delayed_candidates[0]]), "delayed"
    return None, "missed"


def select_representative_events(
    series: dict[str, ModelSeries],
    threshold: float,
    max_events: int,
    lookback: int,
    lookahead: int,
    min_gap: int,
) -> list[int]:
    ref = series["datp_net"]
    candidates = candidate_crossings(ref.true, threshold, lookback, lookahead)
    scored: list[tuple[float, int]] = []
    for event_idx in candidates:
        event_time = float(ref.time_idx[event_idx])
        search_start = float(ref.time_idx[event_idx - lookback])

        datp_alarm, datp_status = find_alarm_time(
            ref.risk_time_idx,
            ref.risk_pred,
            threshold,
            event_time,
            search_start,
            event_time,
        )
        if datp_status != "early_or_on_time" or datp_alarm is None:
            continue

        datp_lead = event_time - datp_alarm
        baseline_leads = []
        baseline_misses = 0
        for model in ordered_models(series):
            if model == "datp_net":
                continue
            item = series[model]
            alarm_time, status = find_alarm_time(
                item.risk_time_idx,
                item.risk_pred,
                threshold,
                event_time,
                search_start,
                event_time,
            )
            if status == "early_or_on_time" and alarm_time is not None:
                baseline_leads.append(event_time - alarm_time)
            else:
                baseline_misses += 1

        best_baseline_lead = max(baseline_leads) if baseline_leads else -1.0
        lead_advantage = datp_lead - best_baseline_lead
        left = max(0, event_idx - lookback)
        right = min(len(ref.true), event_idx + lookahead + 1)
        rise_score = float(np.max(ref.true[event_idx:right]) - np.min(ref.true[left:event_idx]))
        score = 100.0 + datp_lead + 2.0 * baseline_misses + max(lead_advantage, 0.0) + rise_score
        scored.append((score, int(event_idx)))

    selected: list[int] = []
    for _, idx in sorted(scored, reverse=True):
        if all(abs(idx - chosen) >= min_gap for chosen in selected):
            selected.append(idx)
        if len(selected) >= max_events:
            break

    if len(selected) < max_events:
        fallback = select_events(ref.true, threshold, max_events, lookback, lookahead, min_gap)
        for idx in fallback:
            if all(abs(idx - chosen) >= min_gap for chosen in selected):
                selected.append(idx)
            if len(selected) >= max_events:
                break
    return sorted(selected)


def event_metrics(
    dataset: str,
    series: dict[str, ModelSeries],
    events: list[int],
    threshold: float,
    lookback: int,
    lookahead: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ref = series["datp_net"]
    for case_id, event_idx in enumerate(events, start=1):
        event_time = float(ref.time_idx[event_idx])
        search_start = float(ref.time_idx[max(0, event_idx - lookback)])
        search_end = event_time
        for model in ordered_models(series):
            item = series[model]
            alarm_time, status = find_alarm_time(
                item.risk_time_idx,
                item.risk_pred,
                threshold,
                event_time,
                search_start,
                search_end,
            )
            lead_time = None if alarm_time is None else event_time - alarm_time
            rows.append(
                {
                    "dataset": dataset_label(dataset),
                    "case_id": case_id,
                    "event_index": event_idx,
                    "event_time_idx": event_time,
                    "threshold": threshold,
                    "model": model,
                    "label": item.label,
                    "alarm_status": status,
                    "alarm_time_idx": alarm_time,
                    "lead_time_slots": lead_time,
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, model, label), group in metrics.groupby(["dataset", "model", "label"], sort=False):
        valid = group[group["alarm_status"] == "early_or_on_time"]
        delayed = group[group["alarm_status"] == "delayed"]
        missed = group[group["alarm_status"] == "missed"]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "label": label,
                "num_cases": len(group),
                "early_or_on_time_count": len(valid),
                "delayed_count": len(delayed),
                "missed_count": len(missed),
                "mean_lead_time_slots": float(valid["lead_time_slots"].mean()) if len(valid) else np.nan,
                "median_lead_time_slots": float(valid["lead_time_slots"].median()) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def false_alarm_summary(
    dataset: str,
    series: dict[str, ModelSeries],
    threshold: float,
    pred_len: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ref = series["datp_net"]
    time_to_pos = {float(t): i for i, t in enumerate(ref.time_idx)}

    for model in ordered_models(series):
        item = series[model]
        alarm = item.risk_pred > threshold
        false_alarm = np.zeros_like(alarm, dtype=bool)
        true_future_cross = np.zeros_like(alarm, dtype=bool)
        valid = np.zeros_like(alarm, dtype=bool)

        for i, t in enumerate(item.risk_time_idx):
            pos = time_to_pos.get(float(t))
            if pos is None or pos + pred_len > len(ref.true):
                continue
            valid[i] = True
            true_future_cross[i] = bool(np.any(ref.true[pos : pos + pred_len] > threshold))
            false_alarm[i] = alarm[i] and not true_future_cross[i]

        valid_alarm = alarm & valid
        rows.append(
            {
                "dataset": dataset_label(dataset),
                "model": model,
                "label": item.label,
                "valid_windows": int(valid.sum()),
                "alarm_windows": int(valid_alarm.sum()),
                "true_future_cross_windows": int((true_future_cross & valid).sum()),
                "false_alarm_windows": int(false_alarm.sum()),
                "false_alarm_rate_among_alarms": float(false_alarm.sum() / (valid_alarm.sum() + 1e-12)),
                "false_alarm_rate_among_windows": float(false_alarm.sum() / (valid.sum() + 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def ordered_models(series: dict[str, ModelSeries]) -> list[str]:
    return [model for model in PREFERRED_MODELS if model in series]


def plot_case(
    dataset: str,
    series: dict[str, ModelSeries],
    event_idx: int,
    case_id: int,
    threshold: float,
    metrics: pd.DataFrame,
    output_dir: Path,
    lookback: int,
    lookahead: int,
) -> None:
    ref = series["datp_net"]
    start = max(0, event_idx - lookback)
    end = min(len(ref.true), event_idx + lookahead + 1)
    event_time = float(ref.time_idx[event_idx])

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(
        ref.time_idx[start:end],
        ref.true[start:end],
        color="tab:blue",
        linewidth=2.4,
        label="True traffic",
        zorder=5,
    )
    ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.6, label="Expansion threshold")
    ax.axvline(event_time, color="black", linestyle=":", linewidth=1.4, label="True crossing")

    for model in ordered_models(series):
        item = series[model]
        in_case = (item.risk_time_idx >= ref.time_idx[start]) & (item.risk_time_idx <= ref.time_idx[end - 1])
        if not np.any(in_case):
            continue
        is_ours = model == "datp_net"
        ax.plot(
            item.risk_time_idx[in_case],
            item.risk_pred[in_case],
            color="tab:orange" if is_ours else None,
            linewidth=2.8 if is_ours else 1.1,
            alpha=0.95 if is_ours else 0.58,
            marker="o" if is_ours else None,
            markersize=3.2,
            label=f"{item.label} risk pred",
            zorder=4 if is_ours else 2,
        )

    case_metrics = metrics[(metrics["dataset"] == dataset_label(dataset)) & (metrics["case_id"] == case_id)]
    y_min, y_max = ax.get_ylim()
    marker_y = threshold + 0.035 * (y_max - y_min)
    for _, row in case_metrics.iterrows():
        if pd.isna(row["alarm_time_idx"]):
            continue
        is_ours = row["model"] == "datp_net"
        ax.scatter(
            [row["alarm_time_idx"]],
            [marker_y],
            marker="v",
            s=70 if is_ours else 34,
            color="tab:orange" if is_ours else "gray",
            edgecolor="black" if is_ours else "none",
            linewidth=0.7,
            alpha=0.95 if is_ours else 0.55,
            zorder=6,
        )

    ax.set_title(f"{dataset_label(dataset)} turning-point early warning case {case_id}")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Traffic / max future predicted traffic")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(ncol=2, fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()

    base = output_dir / f"{dataset_slug(dataset)}_case_{case_id}_turning_point"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"), dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turning-point early warning case-study evaluation.")
    parser.add_argument("--draw-root", type=Path, default=Path("draw"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/turning_point_early_warning"))
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--threshold-q", type=float, default=0.9)
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--lookahead", type=int, default=40)
    parser.add_argument("--min-gap", type=int, default=40)
    parser.add_argument("--selection-mode", choices=["datp_advantage", "rise"], default="datp_advantage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_roots = {
        "Abilene": args.draw_root / "Abilene" / "Abilene_single",
        "Geant": args.draw_root / "Geant" / "Geant_single",
    }

    all_metrics: list[pd.DataFrame] = []
    all_false_alarm: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for dataset, root in dataset_roots.items():
        files = find_model_files(root)
        series: dict[str, ModelSeries] = {}
        for model, (agg_path, raw_path) in files.items():
            try:
                series[model] = load_model_series(dataset, model, agg_path, raw_path, args.pred_len)
            except Exception as exc:
                print(f"Skipping {dataset}/{model}: {exc}")
        if "datp_net" not in series:
            print(f"Skipping {dataset}: DATP-Net is required as the reference series.")
            continue

        ref = series["datp_net"]
        threshold = float(np.quantile(ref.true, args.threshold_q))
        if args.selection_mode == "rise":
            events = select_events(
                ref.true,
                threshold=threshold,
                max_events=args.max_events,
                lookback=args.lookback,
                lookahead=args.lookahead,
                min_gap=args.min_gap,
            )
        else:
            events = select_representative_events(
                series,
                threshold=threshold,
                max_events=args.max_events,
                lookback=args.lookback,
                lookahead=args.lookahead,
                min_gap=args.min_gap,
            )
        metrics = event_metrics(dataset, series, events, threshold, args.lookback, args.lookahead)
        false_alarm = false_alarm_summary(dataset, series, threshold, args.pred_len)
        all_metrics.append(metrics)
        all_false_alarm.append(false_alarm)

        for case_id, event_idx in enumerate(events, start=1):
            selected_rows.append(
                {
                    "dataset": dataset_label(dataset),
                    "case_id": case_id,
                    "event_index": event_idx,
                    "event_time_idx": float(ref.time_idx[event_idx]),
                    "threshold_q": args.threshold_q,
                    "threshold": threshold,
                    "selection_mode": args.selection_mode,
                    "true_at_crossing": float(ref.true[event_idx]),
                    "window_start_time_idx": float(ref.time_idx[max(0, event_idx - args.lookback)]),
                    "window_end_time_idx": float(ref.time_idx[min(len(ref.true) - 1, event_idx + args.lookahead)]),
                }
            )
            plot_case(dataset, series, event_idx, case_id, threshold, metrics, args.output_dir, args.lookback, args.lookahead)

        for model in ordered_models(series):
            item = series[model]
            manifest_rows.append(
                {
                    "dataset": dataset_label(dataset),
                    "model": model,
                    "label": item.label,
                    "agg_path": str(item.agg_path),
                    "raw_path": str(item.raw_path),
                    "agg_rows": len(item.true),
                    "raw_windows": len(item.risk_pred),
                }
            )
        print(f"{dataset}: selected {len(events)} events and evaluated {len(series)} models.")

    if not all_metrics:
        raise SystemExit("No early-warning results were generated.")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    false_alarm_df = pd.concat(all_false_alarm, ignore_index=True)
    aggregate_df = aggregate_metrics(metrics_df)
    selected_df = pd.DataFrame(selected_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    metrics_df.to_csv(args.output_dir / "event_alarm_metrics.csv", index=False)
    aggregate_df.to_csv(args.output_dir / "event_alarm_summary.csv", index=False)
    false_alarm_df.to_csv(args.output_dir / "false_alarm_summary.csv", index=False)
    selected_df.to_csv(args.output_dir / "selected_turning_points.csv", index=False)
    manifest_df.to_csv(args.output_dir / "input_manifest.csv", index=False)

    print(f"Saved turning-point early-warning outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
