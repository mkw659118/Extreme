from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(r"./draw/extreme_lstm_memo_reservoir_stor_4001_sof24_PL96_DM64.csv")
PRED_LEN = 96
NUM_VARS = 144


def load_vars(var_ids, start_row=0, n_rows=None):
    if isinstance(var_ids, int):
        var_ids = [var_ids]

    usecols = []
    for var_id in var_ids:
        usecols.extend([f"true_var_{var_id}", f"pred_var_{var_id}"])

    return pd.read_csv(
        CSV_PATH,
        usecols=usecols,
        skiprows=range(1, start_row + 1) if start_row > 0 else None,
        nrows=n_rows,
    )


def metrics_np(y_true, y_pred, eps=1e-8):
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "MAE": float(abs_err.mean()),
        "MSE": float((err**2).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAPE": float((abs_err / (np.abs(y_true) + eps)).mean()),
        "NMAE": float(abs_err.sum() / (np.abs(y_true).sum() + eps)),
        "NRMSE": float(np.sqrt((err**2).sum()) / (np.sqrt((y_true**2).sum()) + eps)),
    }


def plot_var(var_id=143, start_row=0, n_points=1200):
    df = load_vars(var_id, start_row=start_row, n_rows=n_points)
    true = df[f"true_var_{var_id}"].to_numpy()
    pred = df[f"pred_var_{var_id}"].to_numpy()
    x = np.arange(start_row, start_row + len(df))
    m = metrics_np(true, pred)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(x, true, label="True", linewidth=1.5)
    axes[0].plot(x, pred, label="Pred", linewidth=1.2, alpha=0.85)
    axes[0].set_title(
        f"Variable {var_id} | "
        f"MAE={m['MAE']:.6g}, RMSE={m['RMSE']:.6g}, NMAE={m['NMAE']:.4f}"
    )
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].plot(x, pred - true, color="tab:red", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Pred - True")
    axes[1].set_xlabel("Flattened row index")

    plt.tight_layout()
    plt.show()


def plot_window(var_id=143, window_id=0):
    plot_var(var_id=var_id, start_row=window_id * PRED_LEN, n_points=PRED_LEN)


def compute_all_var_metrics(chunksize=50_000):
    sum_abs = np.zeros(NUM_VARS, dtype=np.float64)
    sum_sq = np.zeros(NUM_VARS, dtype=np.float64)
    sum_abs_true = np.zeros(NUM_VARS, dtype=np.float64)
    sum_sq_true = np.zeros(NUM_VARS, dtype=np.float64)
    count = 0

    true_names = [f"true_var_{i}" for i in range(NUM_VARS)]
    pred_names = [f"pred_var_{i}" for i in range(NUM_VARS)]

    for chunk in pd.read_csv(CSV_PATH, chunksize=chunksize):
        true = chunk[true_names].to_numpy(dtype=np.float64)
        pred = chunk[pred_names].to_numpy(dtype=np.float64)
        err = pred - true
        sum_abs += np.abs(err).sum(axis=0)
        sum_sq += (err**2).sum(axis=0)
        sum_abs_true += np.abs(true).sum(axis=0)
        sum_sq_true += (true**2).sum(axis=0)
        count += len(chunk)

    return pd.DataFrame(
        {
            "var_id": np.arange(NUM_VARS),
            "MAE": sum_abs / count,
            "MSE": sum_sq / count,
            "RMSE": np.sqrt(sum_sq / count),
            "NMAE": sum_abs / (sum_abs_true + 1e-8),
            "NRMSE": np.sqrt(sum_sq) / (np.sqrt(sum_sq_true) + 1e-8),
            "true_abs_mean": sum_abs_true / count,
        }
    )


def plot_metric_heatmaps(metrics_df):
    mae_matrix = metrics_df.set_index("var_id")["MAE"].to_numpy().reshape(12, 12)
    scale_matrix = metrics_df.set_index("var_id")["true_abs_mean"].to_numpy().reshape(12, 12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(mae_matrix, cmap="viridis")
    axes[0].set_title("MAE by OD pair")
    axes[0].set_xlabel("Destination node")
    axes[0].set_ylabel("Source node")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(scale_matrix, cmap="magma")
    axes[1].set_title("Mean |true| by OD pair")
    axes[1].set_xlabel("Destination node")
    axes[1].set_ylabel("Source node")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH.resolve()}")

    print(f"Reading: {CSV_PATH.resolve()}")

    # 1. Change var_id to inspect another one of the 144 variables.
    plot_var(var_id=143, start_row=0, n_points=1200)

    # 2. Uncomment this to inspect a single forecast window.
    # plot_window(var_id=143, window_id=0)

    # 3. Uncomment this to compute all 144 variable metrics and heatmaps.
    # metrics_df = compute_all_var_metrics()
    # print(metrics_df.sort_values("MAE", ascending=False).head(15))
    # plot_metric_heatmaps(metrics_df)
