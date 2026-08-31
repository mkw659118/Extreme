from __future__ import annotations

from regenerate_mlu_ratio_bigfont import main


OUTPUT_STEM = (
    "Abilene_GEANT_PL5_DM256_mlu_ratio_all_baselines_bold_true_datp_mean_legend_"
    "1x2_updated_600pts_long_true_legend_soft_legend"
)


if __name__ == "__main__":
    main(output_stem=OUTPUT_STEM, legend_frame_alpha=0.62)
