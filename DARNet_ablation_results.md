# DARNet 消融实验结果统计

数据来源：`XR.log`。共统计 5 个模型配置、3 个数据集（Abilene、Geant、Seattle）和 4 个预测步长（5、10、15、20）。

说明：本表主要关注 `NMAE / NRMSE / COS` 三个指标。其中 `NMAE`、`NRMSE` 越低越好，`COS` 越高越好；每个数据集内的最优值已加粗。

## 消融组件设置

| 模型配置 | Missing-aware Encoding | State Prior | Retrieval | MoE |
|---|---:|---:|---:|---:|
| DARNet | 是 | 是 | 是 | 是 |
| w/o MoE | 是 | 是 | 是 | 否 |
| w/o Retrieval | 是 | 是 | 否 | 是 |
| w/o State Prior | 是 | 否 | 是 | 是 |
| w/o Missing-Aware Enc. | 否 | 是 | 是 | 是 |

## Pred Len = 5

| Dataset | Model | NMAE | NRMSE | COS |
|---|---|---:|---:|---:|
| Abilene | DARNet | **0.5198** | **0.5889** | **0.6430** |
|  | w/o MoE | 0.5281 | 0.5897 | 0.6278 |
|  | w/o Retrieval | 0.5230 | 0.5950 | 0.6383 |
|  | w/o State Prior | 0.5234 | 0.5900 | 0.6415 |
|  | w/o Missing-Aware Enc. | 0.5252 | 0.5922 | 0.6406 |
| Geant | DARNet | **0.3893** | **0.5032** | **0.6869** |
|  | w/o MoE | 0.3909 | 0.5036 | 0.6861 |
|  | w/o Retrieval | 0.3902 | 0.5034 | 0.6864 |
|  | w/o State Prior | 0.3895 | 0.5069 | 0.6818 |
|  | w/o Missing-Aware Enc. | 0.3911 | 0.5036 | 0.6861 |
| Seattle | DARNet | **0.5082** | 0.6411 | **0.3651** |
|  | w/o MoE | 0.5111 | 0.6409 | 0.3556 |
|  | w/o Retrieval | 0.5136 | 0.6471 | 0.3476 |
|  | w/o State Prior | 0.5110 | **0.6407** | 0.3557 |
|  | w/o Missing-Aware Enc. | 0.5109 | 0.6416 | 0.3559 |

## Pred Len = 10

| Dataset | Model | NMAE | NRMSE | COS |
|---|---|---:|---:|---:|
| Abilene | DARNet | **0.6133** | 0.6536 | **0.5509** |
|  | w/o MoE | 0.6181 | 0.6523 | 0.5313 |
|  | w/o Retrieval | 0.6143 | **0.6466** | 0.5400 |
|  | w/o State Prior | 0.6229 | 0.6485 | 0.5366 |
|  | w/o Missing-Aware Enc. | 0.6200 | 0.6480 | 0.5385 |
| Geant | DARNet | **0.4323** | **0.5678** | **0.5723** |
|  | w/o MoE | 0.4425 | 0.5696 | 0.5703 |
|  | w/o Retrieval | 0.4416 | 0.5694 | 0.5713 |
|  | w/o State Prior | 0.4394 | 0.5754 | 0.5655 |
|  | w/o Missing-Aware Enc. | 0.4424 | 0.5690 | 0.5714 |
| Seattle | DARNet | **0.5868** | **0.6732** | **0.1359** |
|  | w/o MoE | 0.5870 | 0.6738 | 0.1358 |
|  | w/o Retrieval | 0.5919 | 0.6769 | 0.1309 |
|  | w/o State Prior | 0.5879 | 0.6740 | 0.1354 |
|  | w/o Missing-Aware Enc. | 0.5877 | 0.6739 | 0.1355 |

## Pred Len = 15

| Dataset | Model | NMAE | NRMSE | COS |
|---|---|---:|---:|---:|
| Abilene | DARNet | **0.6651** | 0.6973 | **0.4582** |
|  | w/o MoE | 0.6676 | 0.6950 | 0.4435 |
|  | w/o Retrieval | 0.6723 | 0.6936 | 0.4441 |
|  | w/o State Prior | 0.6766 | 0.6971 | 0.4447 |
|  | w/o Missing-Aware Enc. | 0.6776 | **0.6906** | 0.4533 |
| Geant | DARNet | 0.4776 | **0.6060** | 0.4909 |
|  | w/o MoE | 0.4787 | 0.6064 | 0.4903 |
|  | w/o Retrieval | 0.4780 | 0.6061 | 0.4909 |
|  | w/o State Prior | **0.4721** | 0.6124 | 0.4854 |
|  | w/o Missing-Aware Enc. | 0.4779 | 0.6061 | **0.4916** |
| Seattle | DARNet | **0.5931** | **0.6635** | **0.0746** |
|  | w/o MoE | 0.5938 | 0.6647 | 0.0743 |
|  | w/o Retrieval | 0.5961 | 0.6664 | 0.0722 |
|  | w/o State Prior | 0.5939 | 0.6646 | 0.0743 |
|  | w/o Missing-Aware Enc. | 0.5934 | 0.6649 | 0.0742 |

## Pred Len = 20

| Dataset | Model | NMAE | NRMSE | COS |
|---|---|---:|---:|---:|
| Abilene | DARNet | **0.7032** | 0.7176 | **0.4143** |
|  | w/o MoE | 0.7124 | 0.7156 | 0.3940 |
|  | w/o Retrieval | 0.7120 | **0.7154** | 0.3952 |
|  | w/o State Prior | 0.7100 | 0.7167 | 0.3949 |
|  | w/o Missing-Aware Enc. | 0.7357 | 0.7175 | 0.4001 |
| Geant | DARNet | 0.4939 | **0.6138** | **0.4529** |
|  | w/o MoE | 0.4960 | 0.6222 | 0.4441 |
|  | w/o Retrieval | 0.4939 | 0.6232 | 0.4446 |
|  | w/o State Prior | **0.4894** | 0.6273 | 0.4401 |
|  | w/o Missing-Aware Enc. | 0.4945 | 0.6234 | 0.4431 |
| Seattle | DARNet | **0.6011** | **0.6681** | **0.0117** |
|  | w/o MoE | 0.6028 | 0.6690 | 0.0104 |
|  | w/o Retrieval | 0.6048 | 0.6718 | 0.0054 |
|  | w/o State Prior | 0.6019 | 0.6692 | 0.0105 |
|  | w/o Missing-Aware Enc. | 0.6014 | 0.6690 | 0.0112 |



\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{0.95}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccccccccc}
\hline
\textbf{Model} & \multicolumn{3}{c}{\textbf{Almaden}} & \multicolumn{3}{c}{\textbf{Coyote}} & \multicolumn{3}{c}{\textbf{Lexington}} & \multicolumn{3}{c}{\textbf{Stevens Creek}} & \multicolumn{3}{c}{\textbf{Vasona}} \tabularnewline
 & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} \tabularnewline
\hline
DATP-Net & \textbf{0.0016} & \textbf{0.0062} & \textbf{0.3196} & \textbf{0.0010} & \textbf{0.0041} & \textbf{0.1762} & \textbf{0.0016} & \textbf{0.0209} & \textbf{0.4017} & \textbf{0.0012} & \textbf{0.0038} & \textbf{0.3481} & \textbf{0.0032} & \textbf{0.0113} & 0.1062 \tabularnewline
PMDformer & 0.0036 & 0.0115 & 0.1835 & \underline{0.0014} & 0.0052 & 0.1394 & 0.0028 & 0.0288 & 0.3713 & 0.0025 & 0.0089 & 0.2667 & 0.0038 & 0.0123 & 0.1017 \tabularnewline
HMformer & 0.0026 & 0.0079 & 0.2345 & 0.0018 & 0.0056 & 0.1491 & 0.0031 & 0.0263 & 0.3778 & 0.0018 & 0.0065 & 0.3217 & 0.0040 & 0.0130 & 0.1073 \tabularnewline
FeTS & 0.0053 & 0.0110 & 0.2218 & 0.0060 & 0.0126 & 0.1549 & 0.0070 & 0.0257 & 0.3613 & 0.0058 & 0.0116 & 0.3079 & 0.0054 & 0.0125 & 0.0924 \tabularnewline
TimesNet & 0.0034 & 0.0107 & 0.1960 & 0.0019 & 0.0066 & 0.1551 & 0.0040 & 0.0296 & 0.3498 & 0.0027 & 0.0101 & 0.2828 & 0.0053 & 0.0163 & 0.0575 \tabularnewline
iTransformer & 0.0042 & 0.0105 & 0.2232 & 0.0036 & 0.0079 & \underline{0.1619} & 0.0051 & 0.0276 & \underline{0.3822} & 0.0034 & 0.0089 & 0.3282 & 0.0053 & 0.0146 & \textbf{0.1152} \tabularnewline
FEDformer & 0.0377 & 0.0745 & 0.0275 & 0.0481 & 0.1031 & -0.0856 & 0.0588 & 0.1236 & -0.0112 & 0.0459 & 0.0943 & 0.0003 & 0.0311 & 0.0516 & 0.0216 \tabularnewline
PatchTST & 0.0031 & 0.0083 & 0.2303 & 0.0044 & 0.0099 & 0.1472 & 0.0050 & 0.0276 & 0.3742 & 0.0042 & 0.0093 & 0.3196 & 0.0048 & 0.0127 & 0.1128 \tabularnewline
WPMixer & \underline{0.0025} & 0.0078 & 0.2386 & 0.0025 & 0.0065 & 0.1325 & \underline{0.0027} & \underline{0.0224} & 0.3666 & \underline{0.0017} & 0.0063 & \underline{0.3384} & 0.0040 & 0.0130 & \underline{0.1139} \tabularnewline
P\_sLSTM & 0.0027 & \underline{0.0075} & 0.2405 & 0.0016 & \underline{0.0048} & 0.1592 & 0.0043 & 0.0259 & 0.3743 & 0.0021 & 0.0063 & 0.3196 & \underline{0.0037} & \underline{0.0121} & 0.0903 \tabularnewline
xLSTMTime & 0.0027 & 0.0092 & \underline{0.2508} & \underline{0.0014} & 0.0050 & 0.1453 & 0.0029 & 0.0283 & 0.3686 & \underline{0.0017} & \underline{0.0060} & 0.3245 & 0.0044 & 0.0138 & 0.0952 \tabularnewline
xLSTM-Mixer & 0.0031 & 0.0080 & 0.2188 & 0.0026 & 0.0064 & 0.1115 & 0.0037 & 0.0243 & 0.3665 & 0.0028 & 0.0069 & 0.3349 & 0.0045 & 0.0132 & 0.1052 \tabularnewline
\hline
\end{tabular}%
}
\caption{Comparison of NMAE, NRMSE, and COS on five water datasets under prediction length 8. For DATP-Net, each reported metric is selected independently from all available main-model runs under the same dataset and horizon. Lower NMAE and NRMSE indicate better performance, while higher COS indicates better performance. The best results are highlighted in bold, and the second-best results are underlined.}
\label{tab:water_pred8_baseline_comparison_metricwise_datp}
\end{table*}

\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{0.95}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccccccccc}
\hline
\textbf{Model} & \multicolumn{3}{c}{\textbf{Almaden}} & \multicolumn{3}{c}{\textbf{Coyote}} & \multicolumn{3}{c}{\textbf{Lexington}} & \multicolumn{3}{c}{\textbf{Stevens Creek}} & \multicolumn{3}{c}{\textbf{Vasona}} \tabularnewline
 & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} & \textbf{NMAE} & \textbf{NRMSE} & \textbf{COS} \tabularnewline
\hline
DATP-Net & \textbf{0.0159} & \textbf{0.0511} & \textbf{0.5009} & \textbf{0.0120} & \textbf{0.0466} & \textbf{0.4853} & \textbf{0.0119} & \textbf{0.0570} & \textbf{0.7636} & \underline{0.0115} & \textbf{0.0332} & \textbf{0.6007} & \textbf{0.0181} & \textbf{0.0397} & \textbf{0.1699} \tabularnewline
PMDformer & \underline{0.0161} & \underline{0.0523} & 0.4314 & \underline{0.0123} & 0.0482 & 0.2640 & \underline{0.0130} & 0.0605 & \underline{0.7450} & 0.0123 & 0.0387 & 0.5187 & \underline{0.0189} & 0.0419 & 0.0867 \tabularnewline
HMformer & 0.0182 & 0.0577 & 0.3925 & 0.0136 & 0.0515 & 0.4083 & 0.0154 & 0.0648 & 0.6953 & 0.0120 & 0.0381 & 0.4807 & 0.0209 & 0.0460 & 0.0351 \tabularnewline
FeTS & 0.0176 & 0.0524 & 0.4125 & 0.0133 & \underline{0.0481} & 0.4281 & 0.0164 & 0.0633 & 0.7359 & 0.0139 & 0.0379 & 0.4840 & 0.0202 & 0.0442 & 0.0822 \tabularnewline
TimesNet & 0.0187 & 0.0605 & 0.3646 & 0.0145 & 0.0495 & 0.4334 & 0.0193 & 0.0792 & 0.6668 & 0.0122 & 0.0379 & 0.4978 & 0.0211 & 0.0479 & 0.0124 \tabularnewline
iTransformer & 0.0190 & 0.0598 & 0.4139 & 0.0143 & 0.0502 & 0.4141 & 0.0162 & 0.0658 & 0.7270 & 0.0135 & 0.0395 & 0.5309 & 0.0203 & 0.0439 & 0.1381 \tabularnewline
FEDformer & 0.0488 & 0.0958 & 0.4016 & 0.0621 & 0.1307 & 0.4017 & 0.0651 & 0.1184 & 0.3328 & 0.0558 & 0.1130 & 0.3865 & 0.0409 & 0.0700 & 0.0246 \tabularnewline
PatchTST & 0.0168 & 0.0539 & 0.3900 & 0.0135 & 0.0482 & 0.4313 & 0.0174 & 0.0653 & 0.7136 & 0.0128 & 0.0376 & 0.5196 & 0.0195 & 0.0429 & 0.0810 \tabularnewline
WPMixer & 0.0183 & 0.0581 & 0.4168 & 0.0162 & 0.0595 & 0.3381 & 0.0149 & \underline{0.0584} & 0.7033 & 0.0131 & 0.0424 & \underline{0.5329} & 0.0196 & 0.0429 & 0.1206 \tabularnewline
P\_sLSTM & 0.0190 & 0.0568 & \underline{0.4669} & 0.0149 & 0.0537 & 0.4114 & 0.0203 & 0.0661 & 0.6820 & 0.0142 & 0.0420 & 0.5230 & 0.0215 & 0.0488 & \underline{0.1620} \tabularnewline
xLSTMTime & 0.0176 & 0.0551 & 0.3679 & 0.0143 & 0.0526 & 0.4132 & 0.0144 & 0.0634 & 0.6877 & \textbf{0.0111} & \underline{0.0363} & 0.4922 & 0.0203 & 0.0432 & 0.1008 \tabularnewline
xLSTM-Mixer & 0.0179 & 0.0556 & 0.3733 & 0.0124 & \underline{0.0481} & \underline{0.4433} & 0.0157 & 0.0727 & 0.7352 & 0.0128 & 0.0400 & 0.4941 & 0.0194 & \underline{0.0416} & 0.0560 \tabularnewline
\hline
\end{tabular}%
}
\caption{Comparison of NMAE, NRMSE, and COS on five water datasets under prediction length 72. For DATP-Net, each reported metric is selected independently from all available main-model runs under the same dataset and horizon. Lower NMAE and NRMSE indicate better performance, while higher COS indicates better performance. The best results are highlighted in bold, and the second-best results are underlined.}
\label{tab:water_pred72_baseline_comparison_metricwise_datp}
\end{table*}
