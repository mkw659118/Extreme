# 三组超参数消融实验相对变化图说明

本文档说明 `Abilene_three_hyperparameter_relative_change_NMAE_1x3_coded_scales.pdf` 中三个子图的实验设计、绘图口径和结果解读。该图围绕 DARNet 在 Abilene 数据集上的三组关键超参数展开：专家数量、检索 topK、多尺度状态先验输入。图中展示的是 NMAE 的相对变化，而不是原始 NMAE，因此更适合比较同一组实验内部不同设置相对于基准配置的增益或退化。

## 图文件与数据来源

- 主图文件：`Ablation/figures/combined/Abilene_three_hyperparameter_relative_change_NMAE_1x3_coded_scales.pdf`
- PNG 预览：`Ablation/figures/combined/Abilene_three_hyperparameter_relative_change_NMAE_1x3_coded_scales.png`
- 合并绘图数据：`Ablation/figures/combined/Abilene_three_hyperparameter_relative_change_NMAE_1x3_coded_scales.csv`
- 多尺度先验编码映射表：`Ablation/figures/combined/Abilene_state_prior_scale_code_mapping.csv`
- 绘图脚本：`Ablation/plot_three_hyperparameter_relative_change_abilene_nmae_1x3.py`

三个子图均基于 `Ablation/parsed_hyperparameter_ablation_results_aggregated.csv` 中的聚合测试结果生成。评价指标为 NMAE，预测步长包含 `PredLen=5` 和 `PredLen=20`。由于 NMAE 是误差类指标，数值越低表示预测效果越好。因此，在相对变化图中，曲线低于 0 表示该配置相较基准配置带来了性能提升；曲线高于 0 表示该配置相较基准配置出现了性能退化。

## 统一绘图口径

三个子图均采用相同的相对变化公式：

```text
Relative Change (%) = (NMAE_current - NMAE_baseline) / NMAE_baseline * 100
```

其中 `NMAE_current` 表示当前超参数配置下的 NMAE，`NMAE_baseline` 表示该组实验内部选定的基准配置对应的 NMAE。每个预测步长单独计算自己的基准值，即 `PredLen=5` 和 `PredLen=20` 不共用同一个分母。

这样处理有两个好处。第一，可以消除不同预测步长本身误差量级不同带来的影响，使曲线反映“相对基准的变化趋势”。第二，可以更清楚地观察某个超参数对短期预测和较长步长预测的影响是否一致。

需要注意的是，三个子图的纵轴是独立缩放的，不能直接用视觉高度比较不同子图之间的绝对影响强度。例如检索 topK 子图的相对变化幅度很小，只有约 `-0.22%` 到 `0.15%`；而专家数量和多尺度先验子图的变化幅度更明显。因此，跨子图比较时应看具体数值，而不是只看线条起伏的视觉大小。

## 子图 (a)：专家数量实验

该实验考察 MoE 中专家数量对模型性能的影响。实验变量为 `Num_Experts`，取值为 `1, 2, 4, 6, 8`。其他主要配置保持不变：`Seq_Len=96`，`Retrieval_Num=2`，`State_Num=4`，`State_Prior_Setting=1,4,8,16`，`Use_Retrieval=True`，`Use_State_Prior=True`，`Use_Missing_Aware_Encoding=True`。

该组实验的基准配置为 `Num_Experts=1`。当专家数量为 1 时，模型退化为单专家形式，不再具有多专家之间的分工与选择能力。因此，该子图可以用于回答“多专家机制是否比单专家更有效”以及“专家数量继续增加是否持续有益”这两个问题。

具体结果如下：

| Num_Experts | PredLen=5 NMAE | PredLen=5 相对变化 | PredLen=20 NMAE | PredLen=20 相对变化 |
|---:|---:|---:|---:|---:|
| 1 | 0.52693231 | 0.0000% | 0.71635656 | 0.0000% |
| 2 | 0.52379856 | -0.5947% | 0.71763432 | +0.1784% |
| 4 | 0.52249281 | -0.8425% | 0.70823435 | -1.1338% |
| 6 | 0.53634538 | +1.7864% | 0.71658156 | +0.0314% |
| 8 | 0.53855442 | +2.2056% | 0.72234522 | +0.8360% |

从图中可以看出，`Num_Experts=4` 是该组设置中的最优点。对于 `PredLen=5`，4 个专家相比单专家带来了约 `0.8425%` 的 NMAE 降低；对于 `PredLen=20`，4 个专家带来了约 `1.1338%` 的 NMAE 降低。这个结果说明，多专家结构相较单专家结构确实能够提升预测性能，尤其在较长预测步长上收益更明显。

同时，专家数量并不是越多越好。当专家数增加到 6 或 8 时，`PredLen=5` 和 `PredLen=20` 的性能均出现不同程度退化。一个合理解释是，在当前数据规模和任务设置下，过多专家会增加路由和专家分工的不稳定性，也可能导致每个专家获得的有效训练样本减少，从而削弱泛化能力。因此，该实验支持使用适中的专家数量，而不是盲目扩大 MoE 容量。

## 子图 (b)：检索 topK 实验

该实验考察检索分支中检索样本数量对预测性能的影响。实验变量为 `Retrieval_Num`，取值为 `1, 2, 3, 5, 8`。其他主要配置保持不变：`Seq_Len=96`，`State_Num=4`，`Num_Experts=4`，`Top_K_Experts=2`，`State_Prior_Setting=1,4,8,16`，`Use_Retrieval=True`，`Use_State_Prior=True`，`Use_Missing_Aware_Encoding=True`。

该组实验的基准配置为 `Retrieval_Num=1`。因此，子图 (b) 描述的是从只检索 1 个相似样本开始，逐步增加检索数量后模型性能相对基准的变化。

具体结果如下：

| Retrieval topK | PredLen=5 NMAE | PredLen=5 相对变化 | PredLen=20 NMAE | PredLen=20 相对变化 |
|---:|---:|---:|---:|---:|
| 1 | 0.52363516 | 0.0000% | 0.70866358 | 0.0000% |
| 2 | 0.52249281 | -0.2182% | 0.70823435 | -0.0606% |
| 3 | 0.52432337 | +0.1314% | 0.70879588 | +0.0187% |
| 5 | 0.52442508 | +0.1509% | 0.70886489 | +0.0284% |
| 8 | 0.52427215 | +0.1216% | 0.70855956 | -0.0147% |

从结果看，`Retrieval_Num=2` 是该组实验中最稳定且整体最优的配置。对于 `PredLen=5`，topK 从 1 增加到 2 后 NMAE 相对下降约 `0.2182%`；对于 `PredLen=20`，相对下降约 `0.0606%`。虽然检索 topK 的整体影响幅度不如专家数量明显，但其趋势仍表明，适量引入相似历史片段能够提供有用的上下文信息。

当 topK 继续增加到 3、5 或 8 时，短期预测 `PredLen=5` 的误差反而略有上升。这说明检索数量过多时，额外引入的样本可能包含弱相关或噪声信息，导致检索分支提供的参考上下文不再足够精确。对于 `PredLen=20`，topK 变化带来的差异更小，说明较长步长预测对检索数量的敏感性较弱，或者检索分支的收益在较长预测范围内被其他模块的建模误差部分抵消。

因此，子图 (b) 更适合支撑这样的结论：检索分支并非简单地“检索越多越好”，而是需要选择一个适中的相似样本数量；在当前实验中，`Retrieval_Num=2` 是较合理的折中点。

## 子图 (c)：多尺度状态先验输入实验

该实验考察 Student-t 状态先验或状态先验路由输入中使用的多尺度统计信息是否有助于模型性能。实验变量为 `State_Prior_Setting`，即输入到状态先验分支的尺度组合。为了避免横坐标过长，图中使用 `A-F` 对不同尺度组合进行编码。

编码关系如下：

| Code | State_Prior_Setting | 含义 |
|---:|---|---|
| A | `1` | 仅使用最短尺度状态先验输入 |
| B | `1,4` | 使用尺度 1 和 4 |
| C | `1,4,8` | 使用尺度 1、4 和 8 |
| D | `1,4,8,16` | 使用尺度 1、4、8 和 16 |
| E | `1,4,8,16,32` | 使用尺度 1、4、8、16 和 32 |
| F | `1,4,8,16_no_seq` | 使用尺度 1、4、8、16，但关闭序列级先验输入 |

其他主要配置保持不变：`Seq_Len=96`，`Retrieval_Num=2`，`State_Num=4`，`Num_Experts=4`，`Top_K_Experts=2`，`Use_Retrieval=True`，`Use_State_Prior=True`，`Use_Missing_Aware_Encoding=True`。该组实验的基准配置为 `A`，即仅使用单尺度 `1`。

具体结果如下：

| Code | State_Prior_Setting | PredLen=5 NMAE | PredLen=5 相对变化 | PredLen=20 NMAE | PredLen=20 相对变化 |
|---:|---|---:|---:|---:|---:|
| A | `1` | 0.52242134 | 0.0000% | 0.71169686 | 0.0000% |
| B | `1,4` | 0.52495907 | +0.4858% | 0.71697485 | +0.7416% |
| C | `1,4,8` | 0.52774890 | +1.0198% | 0.71465487 | +0.4156% |
| D | `1,4,8,16` | 0.52249281 | +0.0137% | 0.70823435 | -0.4865% |
| E | `1,4,8,16,32` | 0.52142485 | -0.1907% | 0.71719253 | +0.7722% |
| F | `1,4,8,16_no_seq` | 0.53358737 | +2.1374% | 0.71077032 | -0.1302% |

该子图显示，多尺度状态先验对不同预测步长的影响并不完全一致。对于 `PredLen=5`，最佳配置是 `E=1,4,8,16,32`，相对单尺度基准 NMAE 下降约 `0.1907%`。这说明在短期预测中，加入更长尺度的先验输入可能提供了额外的局部-全局状态信息，有助于模型更准确地识别当前样本所处的状态。

对于 `PredLen=20`，最佳配置是 `D=1,4,8,16`，相对单尺度基准 NMAE 下降约 `0.4865%`。相比之下，继续加入尺度 32 后，`PredLen=20` 的误差反而明显上升。这表明过长尺度的信息并不一定有利于较长预测步长，可能因为过长窗口平滑了局部异常或短期变化，使状态先验输入对当前预测目标的指示性下降。

另外，`F=1,4,8,16_no_seq` 用于考察序列级先验输入的作用。该配置在 `PredLen=5` 上出现最明显退化，相对基准上升约 `2.1374%`，说明序列级先验对短期预测中的状态识别尤其重要。但在 `PredLen=20` 上，关闭序列级输入反而比单尺度基准略好，说明序列级先验对不同预测范围的贡献可能存在差异。这一现象提示后续可以进一步分析序列级状态先验与预测步长之间的关系。

## 围绕三张子图的总体结论

从三组实验可以得到以下结论：

1. 专家数量实验表明，MoE 的多专家结构是有效的，但专家数量需要适中。`Num_Experts=4` 在 `PredLen=5` 和 `PredLen=20` 上均取得最优结果，说明 4 个专家能够较好地平衡模型容量、专家分工和训练稳定性。

2. 检索 topK 实验表明，检索分支对性能有一定增益，但检索数量不是越大越好。`Retrieval_Num=2` 是整体较优配置，说明少量高相关历史片段比大量可能含噪的检索片段更有效。

3. 多尺度状态先验实验表明，状态先验输入的尺度组合会影响模型性能，而且短期预测和较长步长预测对尺度的偏好不同。短期预测更受益于包含尺度 32 的更丰富多尺度输入，而 `PredLen=20` 更偏好 `1,4,8,16` 的中等尺度组合。

4. 三个子图共同说明，DARNet 的关键结构不是单一模块独立起作用，而是由检索增强、状态先验和 MoE 专家分工共同影响性能。合理的超参数组合可以提升预测效果，但过多检索样本、过多专家或不合适的先验尺度都会引入额外噪声或训练不稳定性。

## 可用于论文中的图注草稿

可以将该图描述为：

```text
Figure X. Relative NMAE changes of DARNet under three hyperparameter settings on the Abilene dataset. 
(a) Effect of the number of experts in the MoE module, using one expert as the baseline. 
(b) Effect of retrieval topK, using topK=1 as the baseline. 
(c) Effect of multi-scale state prior inputs, using the single-scale setting A as the baseline. 
Negative values indicate improvements over the corresponding baseline. 
The codes A-F denote different state-prior scale combinations: A=1, B=1,4, C=1,4,8, D=1,4,8,16, E=1,4,8,16,32, and F=1,4,8,16 without sequence-level prior input.
```

如果正文使用中文，可以写为：

```text
图 X 展示了 DARNet 在 Abilene 数据集上三组关键超参数的相对 NMAE 变化。
子图 (a) 分析 MoE 专家数量的影响，并以单专家作为基准；
子图 (b) 分析检索分支中 topK 的影响，并以 topK=1 作为基准；
子图 (c) 分析状态先验输入的多尺度组合影响，并以单尺度设置 A 作为基准。
图中负值表示相对于对应基准配置取得了更低的 NMAE。
A-F 分别表示不同的状态先验尺度组合，其中 A=1，B=1,4，C=1,4,8，D=1,4,8,16，E=1,4,8,16,32，F=1,4,8,16 且不使用序列级先验输入。
```

## 写作时需要注意的点

第一，图中的相对变化是针对每个子图内部基准计算的，不适合直接说“某个子图比另一个子图提升更大”，除非同时引用具体数值。尤其是检索 topK 的变化幅度较小，图中为了可读性使用了独立纵轴，因此视觉波动不能直接和其他子图比较。

第二，`PredLen=5` 和 `PredLen=20` 的基准 NMAE 不同，因此它们的相对变化也是分别归一化的。写作时应避免把两个预测步长的相对变化解释成同一个绝对误差空间中的变化。

第三，`F=1,4,8,16_no_seq` 不只是一个普通尺度组合，它还关闭了序列级先验输入。因此在解释子图 (c) 时，应单独说明 F 的含义，不能把它简单看作 `D` 的另一个尺度变体。

第四，专家数量实验中 `Num_Experts=1` 同时对应 `Top_K_Experts=1`，而其他多专家配置使用 `Top_K_Experts=2`。这符合单专家退化设置的逻辑，但在写作时可以说明该基准表示单专家模型，用于衡量多专家机制本身的贡献。

## 推荐正文表述

综合三个子图，可以在正文中写成：

```text
To further analyze the sensitivity of DARNet to key hyperparameters, we report the relative NMAE changes on the Abilene dataset under different numbers of experts, retrieval topK values, and state-prior scale settings. 
The results show that using four experts consistently improves both short-term and longer-horizon prediction compared with the single-expert baseline, while further increasing the number of experts leads to performance degradation. 
For the retrieval branch, topK=2 achieves the best overall performance, indicating that a small number of highly relevant historical patterns is more beneficial than retrieving more potentially noisy candidates. 
For the state-prior input, the optimal scale combination depends on the prediction horizon: the short horizon benefits from the broader multi-scale setting including scale 32, whereas the longer horizon prefers the 1,4,8,16 configuration. 
These observations suggest that DARNet benefits from moderate model capacity, compact retrieval context, and carefully selected multi-scale state-prior information.
```

中文版本可以写成：

```text
为了进一步分析 DARNet 对关键超参数的敏感性，我们在 Abilene 数据集上比较了不同专家数量、检索 topK 以及状态先验尺度组合下的相对 NMAE 变化。
结果表明，4 个专家相较单专家基准在短期和较长步长预测中均能取得更低的 NMAE，而继续增加专家数量会造成性能退化，说明适中的专家容量更有利于专家分工和泛化。
对于检索分支，topK=2 整体表现最优，说明少量高相关历史模式比引入更多潜在噪声样本更有效。
对于状态先验输入，不同预测步长对尺度组合的偏好不同：短期预测更受益于包含尺度 32 的更丰富多尺度输入，而较长步长预测更适合使用 1,4,8,16 的中等尺度组合。
这些结果说明，DARNet 的性能依赖于模型容量、检索上下文和状态先验尺度之间的平衡。
```
