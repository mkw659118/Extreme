from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from scipy import optimize, stats


FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

PATCH_STYLES = [
    {"fill": "#EAF1FE", "line": "#5477C4", "edge": "#2E4780"},
    {"fill": "#FFF4C2", "line": "#B8A037", "edge": "#736422"},
    {"fill": "#D8ECBD", "line": "#71B436", "edge": "#386411"},
    {"fill": "#FCDAD6", "line": "#BD569B", "edge": "#8A3A6F"},
]


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.65,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def format_compact_value(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1000.0:
        return f"{value:,.0f}"
    if abs_value >= 1.0:
        return f"{value:.2f}"
    if abs_value >= 0.01:
        return f"{value:.4f}"
    return f"{value:.5f}"


def load_long_table_as_matrix(path: Path) -> tuple[np.ndarray, int, int]:
    raw = np.load(path)
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError(f"Expected long table [src, dst, time, value], got {raw.shape}.")

    src = raw[:, 0].astype(np.int64)
    dst = raw[:, 1].astype(np.int64)
    time = raw[:, 2].astype(np.int64)
    value = raw[:, 3].astype(np.float64)

    n_src = int(src.max()) + 1
    n_dst = int(dst.max()) + 1
    length = int(time.max()) + 1

    matrix = np.zeros((length, n_src * n_dst), dtype=np.float64)
    matrix[time, src * n_dst + dst] = value
    return matrix, n_src, n_dst


def fit_student_t_bounded(
    x: np.ndarray,
    min_df: float = 2.1,
    max_df: float = 30.0,
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("Each patch needs at least three finite points.")

    loc0 = float(np.mean(x))
    scale0 = float(np.std(x, ddof=0))
    if not np.isfinite(scale0) or scale0 <= 0:
        scale0 = max(float(np.mean(np.abs(x))) * 0.01, 1e-6)

    def nll(params: np.ndarray) -> float:
        df, loc, log_scale = params
        scale = math.exp(float(log_scale))
        if not np.isfinite(scale) or scale <= 0:
            return float("inf")
        loss = -np.sum(stats.t.logpdf(x, df=df, loc=loc, scale=scale))
        return float(loss) if np.isfinite(loss) else float("inf")

    loc_pad = max(scale0 * 8.0, 1e-6)
    lower_scale = max(scale0 * 1e-3, 1e-12)
    upper_scale = max(scale0 * 100.0, 1e-9)
    bounds = [
        (float(min_df), float(max_df)),
        (float(x.min() - loc_pad), float(x.max() + loc_pad)),
        (math.log(lower_scale), math.log(upper_scale)),
    ]

    best = None
    for df0 in [min_df, 3.0, 5.0, 8.0, 15.0, max_df]:
        for scale_mult in [0.5, 1.0, 2.0]:
            res = optimize.minimize(
                nll,
                np.array([df0, loc0, math.log(scale0 * scale_mult)], dtype=np.float64),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000},
            )
            if best is None or res.fun < best.fun:
                best = res

    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Student-t fit failed.")

    df, loc, log_scale = best.x
    return float(df), float(loc), float(math.exp(log_scale))


def select_default_window(
    data: np.ndarray,
    n_dst: int,
    train_end: int,
    seq_len: int,
    stride: int,
) -> tuple[int, int]:
    candidates = []
    starts = range(0, train_end - seq_len + 1, stride)
    patch_len = seq_len // 4

    for channel in range(data.shape[1]):
        src, dst = divmod(channel, n_dst)
        if src == dst:
            continue

        for start in starts:
            window = data[start : start + seq_len, channel]
            valid = np.isfinite(window) & (window != 0)
            if valid.mean() < 0.98:
                continue

            patch_means = []
            patch_stds = []
            ok = True
            for patch_id in range(4):
                patch = window[patch_id * patch_len : (patch_id + 1) * patch_len]
                patch = patch[np.isfinite(patch) & (patch != 0)]
                if patch.size < patch_len * 0.85:
                    ok = False
                    break
                patch_std = float(np.std(patch, ddof=0))
                if patch_std < 1e-4:
                    ok = False
                    break
                patch_means.append(float(np.mean(patch)))
                patch_stds.append(patch_std)

            if not ok:
                continue

            means = np.array(patch_means, dtype=np.float64)
            stds = np.array(patch_stds, dtype=np.float64)
            center_shift = float(np.std(means) / (np.mean(stds) + 1e-12))
            balance = float(np.min(stds) / (np.max(stds) + 1e-12))
            amplitude = float(np.log(np.mean(means) + 1e-9))
            score = center_shift + balance + 0.15 * amplitude
            candidates.append((score, channel, start))

    if not candidates:
        raise RuntimeError("Could not find a valid off-diagonal training window.")

    _, channel, start = max(candidates, key=lambda item: item[0])
    return int(channel), int(start)


def plot_mapping(
    window: np.ndarray,
    patches: list[np.ndarray],
    params: list[tuple[float, float, float]],
    output_pdf: Path,
    output_png: Path,
    output_svg: Path,
    dataset_name: str,
    start: int,
    channel: int,
    src: int,
    dst: int,
    stride: int,
) -> None:
    use_chart_theme()

    fig = plt.figure(figsize=(7.4, 3.25), dpi=220)
    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1.05, 1.0],
        hspace=0.62,
        wspace=0.24,
    )

    ax_top = fig.add_subplot(gs[0, :])
    x = np.arange(window.size)
    patch_len = patches[0].size
    y_pad = max((window.max() - window.min()) * 0.12, 1e-6)
    y_low = float(window.min() - y_pad)
    y_high = float(window.max() + y_pad)

    for idx, style in enumerate(PATCH_STYLES):
        left = idx * patch_len
        right = (idx + 1) * patch_len - 1
        patch_mean = float(np.mean(window[left : right + 1]))
        patch_y_pos = 0.14 if (patch_mean - y_low) / (y_high - y_low) > 0.58 else 0.88
        ax_top.axvspan(left, right, color=style["fill"], alpha=0.86, lw=0)
        ax_top.text(
            left + patch_len / 2.0 - 0.5,
            patch_y_pos,
            f"Patch {idx + 1}",
            transform=ax_top.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=8.0,
            color=TOKENS["ink"],
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": TOKENS["panel"],
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )

    ax_top.plot(x, window, color=TOKENS["ink"], lw=1.25, zorder=3)
    ax_top.scatter(x[::4], window[::4], s=6, color=TOKENS["ink"], zorder=4)
    ax_top.set_xlim(-1, window.size)
    ax_top.set_ylim(y_low, y_high)
    ax_top.set_xticks([0, 24, 48, 72, 95])
    ax_top.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax_top.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: format_compact_value(value))
    )
    ax_top.set_xlabel("")
    ax_top.set_ylabel("Value", fontsize=8.0, color=TOKENS["muted"], labelpad=4)
    ax_top.grid(axis="y", color=TOKENS["grid"], linewidth=0.65)
    ax_top.spines["left"].set_visible(True)
    ax_top.spines["left"].set_color(TOKENS["axis"])
    ax_top.tick_params(axis="x", labelsize=7.2, length=0, colors=TOKENS["muted"])
    ax_top.tick_params(axis="y", labelsize=7.0, length=0, colors=TOKENS["muted"])
    ax_top.text(
        0.0,
        1.08,
        f"{dataset_name} training window sample",
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        color=TOKENS["muted"],
    )

    bottom_axes = []
    global_min = min(float(np.min(p)) for p in patches)
    global_max = max(float(np.max(p)) for p in patches)
    x_pad = max((global_max - global_min) * 0.18, 1e-6)
    xs_global = np.linspace(global_min - x_pad, global_max + x_pad, 420)

    for idx, (patch, fit_params, style) in enumerate(zip(patches, params, PATCH_STYLES)):
        ax = fig.add_subplot(gs[1, idx])
        bottom_axes.append(ax)
        df, mu, sigma = fit_params

        ys = stats.t.pdf(xs_global, df=df, loc=mu, scale=sigma)
        ax.plot(xs_global, ys, color=style["line"], lw=1.35)
        ax.fill_between(xs_global, 0, ys, color=style["fill"], alpha=0.55, linewidth=0)
        ax.set_xlim(global_min - x_pad, global_max + x_pad)
        ax.set_ylim(bottom=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(TOKENS["axis"])
            ax.spines[spine].set_linewidth(0.6)

        ax.text(
            0.5,
            -0.16,
            f"Student-t {idx + 1}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.8,
            color=TOKENS["ink"],
        )
        ax.text(
            0.5,
            -0.34,
            rf"$\mu$={format_compact_value(mu)}, $\sigma$={format_compact_value(sigma)}, df={df:.1f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.4,
            color=TOKENS["muted"],
        )

    for idx, ax in enumerate(bottom_axes):
        left = idx * patch_len + patch_len / 2.0 - 0.5
        con = ConnectionPatch(
            xyA=(left, ax_top.get_ylim()[0]),
            coordsA=ax_top.transData,
            xyB=(0.5, 1.08),
            coordsB=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=0.8,
            color="#5D7080",
            shrinkA=4,
            shrinkB=2,
        )
        fig.add_artist(con)

    fig.subplots_adjust(left=0.035, right=0.99, top=0.91, bottom=0.23)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_svg, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/Abilene/Abilene_12_12_3000.npy"))
    parser.add_argument("--dataset-name", default="Abilene")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--patches", type=int, default=4)
    parser.add_argument("--channel", type=int, default=122)
    parser.add_argument("--start", type=int, default=1824)
    parser.add_argument("--auto-select", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    if args.seq_len % args.patches != 0:
        raise ValueError("seq-len must be divisible by patches.")

    data, _, n_dst = load_long_table_as_matrix(args.data)
    train_end = int(data.shape[0] * 0.7)

    if args.auto_select:
        channel, start = select_default_window(
            data=data,
            n_dst=n_dst,
            train_end=train_end,
            seq_len=args.seq_len,
            stride=args.stride,
        )
    else:
        channel = int(args.channel)
        start = int(args.start)

    if start % args.stride != 0:
        raise ValueError(f"start={start} is not aligned to stride={args.stride}.")
    if start < 0 or start + args.seq_len > train_end:
        raise ValueError(
            f"Window [{start}, {start + args.seq_len}) is outside training split [0, {train_end})."
        )

    src, dst = divmod(channel, n_dst)
    window = data[start : start + args.seq_len, channel].astype(np.float64)
    if not np.isfinite(window).all():
        raise ValueError("Selected window contains non-finite values.")

    patch_len = args.seq_len // args.patches
    patches = [
        window[i * patch_len : (i + 1) * patch_len].copy()
        for i in range(args.patches)
    ]
    params = [fit_student_t_bounded(patch) for patch in patches]

    output_pdf = args.output_dir / "patch_student_t_mapping_real_data.pdf"
    output_png = args.output_dir / "patch_student_t_mapping_real_data.png"
    output_svg = args.output_dir / "patch_student_t_mapping_real_data.svg"
    main_pdf = args.output_dir / "patch_student_t_mapping.pdf"

    plot_mapping(
        window=window,
        patches=patches,
        params=params,
        output_pdf=output_pdf,
        output_png=output_png,
        output_svg=output_svg,
        dataset_name=args.dataset_name,
        start=start,
        channel=channel,
        src=src,
        dst=dst,
        stride=args.stride,
    )
    main_pdf.write_bytes(output_pdf.read_bytes())

    print(f"data={args.data}")
    print(f"matrix_shape={data.shape}, train_end={train_end}")
    print(f"start={start}, seq_len={args.seq_len}, stride={args.stride}")
    print(f"channel={channel}, src={src}, dst={dst}")
    for idx, (patch, (df, mu, sigma)) in enumerate(zip(patches, params), start=1):
        print(
            f"patch{idx}: n={patch.size}, mean={patch.mean():.8f}, "
            f"std={patch.std(ddof=0):.8f}, mu={mu:.8f}, "
            f"sigma={sigma:.8f}, df={df:.4f}"
        )
    print(f"wrote={output_pdf}")
    print(f"wrote={main_pdf}")


if __name__ == "__main__":
    main()
