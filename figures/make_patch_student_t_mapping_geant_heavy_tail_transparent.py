from __future__ import annotations

from pathlib import Path

import numpy as np

from make_patch_student_t_mapping_real_data import (
    fit_student_t_bounded,
    load_long_table_as_matrix,
    plot_mapping,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "datasets" / "Geant" / "Geant_23_23_3000.npy"
OUTPUT_DIR = ROOT / "figures"

SEQ_LEN = 96
PATCHES = 4
STRIDE = 96
CHANNEL = 415
START = 1632


def safe_print_path(label: str, path: Path) -> None:
    text = str(path).encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"{label}={text}")


def main() -> None:
    data, _, n_dst = load_long_table_as_matrix(DATA_PATH)
    src, dst = divmod(CHANNEL, n_dst)
    window = data[START : START + SEQ_LEN, CHANNEL].astype(np.float64)
    patch_len = SEQ_LEN // PATCHES
    patches = [
        window[i * patch_len : (i + 1) * patch_len].copy()
        for i in range(PATCHES)
    ]
    params = [fit_student_t_bounded(patch) for patch in patches]

    stem = "patch_student_t_mapping_G\u00c9ANT_heavy_tail_df_small_transparent"
    output_pdf = OUTPUT_DIR / f"{stem}.pdf"
    output_png = OUTPUT_DIR / f"{stem}.png"
    output_svg = OUTPUT_DIR / f"{stem}.svg"

    plot_mapping(
        window=window,
        patches=patches,
        params=params,
        output_pdf=output_pdf,
        output_png=output_png,
        output_svg=output_svg,
        dataset_name="G\u00c9ANT heavy-tail",
        start=START,
        channel=CHANNEL,
        src=src,
        dst=dst,
        stride=STRIDE,
        bold_fonts=True,
        bottom_text_inside=True,
        transparent_background=True,
    )

    safe_print_path("wrote", output_pdf)
    safe_print_path("wrote", output_png)
    safe_print_path("wrote", output_svg)


if __name__ == "__main__":
    main()
