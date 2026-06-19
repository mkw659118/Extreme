# PL5 Distribution Similarity Compact Table

Lower values indicate that the prediction distribution is closer to the ground-truth distribution.

| Model        | Abilene Wasserstein/IQR | Abilene JS Distance | GÉANT Wasserstein/IQR | GÉANT JS Distance |
| ------------ | ----------------------- | ------------------- | --------------------- | ----------------- |
| DATP-Net     | 0.1020                  | 0.2347              | 0.0617                | 0.1807            |
| PMDformer    | 0.1789                  | 0.2884              | 0.1307                | 0.1878            |
| HMformer     | 0.2690                  | 0.3318              | 0.1103                | 0.2294            |
| FeTS         | 0.2561                  | 0.3603              | 0.1163                | 0.2053            |
| TimesNet     | 0.1460                  | 0.3347              | 0.1660                | 0.1840            |
| iTransformer | 0.1727                  | 0.3420              | 0.1184                | 0.1773            |
| FEDformer    | 0.2856                  | 0.4295              | 0.1341                | 0.2304            |
| PatchTST     | 0.1970                  | 0.2794              | 0.1372                | 0.1811            |
| WPMixer      | 0.2043                  | 0.3020              | 0.1628                | 0.2015            |
| P-sLSTM      | 0.2927                  | 0.4168              | 0.2362                | 0.3345            |
| xLSTMTime    | 0.2250                  | 0.3945              | 0.1323                | 0.2614            |
| xLSTM-Mixer  | 0.1536                  | 0.2775              | 0.1280                | 0.1908            |

Metrics: `Wasserstein/IQR` is normalized by the ground-truth IQR; `JS Distance` is Jensen-Shannon distance on common histogram bins.
Scope: Abilene and GÉANT, pred_len=5, d_model=256, target_col=TC0.