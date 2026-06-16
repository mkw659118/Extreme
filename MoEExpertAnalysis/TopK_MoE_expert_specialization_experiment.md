# Top-K MoE 专家专门化可视化实验

## 实验目的

本实验用于检查 DATP-Net 中的 Top-K MoE 结构是否真的形成了专家专门化。

具体来说，不只看模型指标是否提升，而是检查 router 是否会把不同动态模式、不同状态组件、不同波动强度的测试样本分配给不同专家。

## 实验设置

- 数据集：Abilene、Geant
- 预测步长：Pred Len = 5
- 分析 split：测试集
- 专家数量：4
- Top-K：2
- Checkpoint round：0
- 模型 checkpoint：
  - `checkpoints/net/DatasetAbilene_Modelnet_PL5_DM256_BS32_d849a064_round_0.pt`
  - `checkpoints/net/DatasetGeant_Modelnet_PL5_DM256_BS32_ae4eaf94_round_0.pt`

脚本会从 checkpoint 自动检测输入通道数。本次使用的 PL5 checkpoint 输入维度为 3，对应特征顺序为：

```text
[diff_norm, second_diff_raw, raw]
```

也就是说，这批 checkpoint 不是 6 通道 missing-aware 输入版本。

## 收集的路由信号

测试集前向传播时，对每个样本保存以下信息：

- `router_prob`：每个样本分配到 4 个专家的概率。
- `topk_experts`：Top-K 路由选中的专家编号。
- `state_probs`：Student-T state prior 输出的状态责任分布。
- `top1_expert`：router 概率最大的专家。
- `routing_entropy`：专家分布熵，越低表示路由越尖锐。
- 输入/未来模式特征：
  - `input_std`
  - `abs_diff_max`
  - `future_abs_diff_max`
  - `missing_rate`

## 生成文件

### 图像

- [Expert Usage](figures/moe_expert_usage_PL5.pdf)
- [State-Expert Alignment](figures/moe_state_expert_alignment_PL5.pdf)
- [Feature by Top-K Expert](figures/moe_feature_by_topk_expert_PL5.pdf)
- [Representative Samples](figures/moe_representative_samples_PL5.pdf)

对应 PNG 也保存在同一目录：

```text
MoEExpertAnalysis/figures/
```

### 数据表

- [所有测试样本 routing 明细](data/moe_routing_samples_all.csv)
- [专家使用率](data/moe_expert_usage.csv)
- [状态-专家对齐矩阵](data/moe_state_expert_alignment.csv)
- [汇总指标](data/moe_specialization_summary.csv)

## 汇总结果

| Dataset | Pred Len | Test Samples | Mean Routing Entropy | Expert-State MI | Dominant Top-1 Expert Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Abilene | 5 | 596 | 0.9578 | 0.0000 | 1.0000 |
| Geant | 5 | 596 | 0.9693 | 0.0000 | 1.0000 |

## 关键发现

### 1. Top-1 路由发生了明显塌缩

在 Abilene 和 Geant 上，`dominant_top1_expert_ratio = 1.0`。

这意味着测试集中所有样本的 Top-1 expert 都是同一个专家，即 Expert 0。

因此，如果用 Top-1 分配来判断专家专门化，本轮 checkpoint 没有形成清晰的多专家分工。

### 2. Top-K 层面仍然有次级专家参与

虽然 Top-1 全部是 Expert 0，但 Top-K 使用率显示第二个专家仍然参与：

Abilene：

```text
Expert 0: Top-K usage = 1.000
Expert 1: Top-K usage = 0.431
Expert 2: Top-K usage = 0.334
Expert 3: Top-K usage = 0.235
```

Geant：

```text
Expert 0: Top-K usage = 1.000
Expert 3: Top-K usage = 1.000
Expert 1/2: Top-K usage = 0.000
```

这说明 Top-K 机制并非完全只使用一个专家；但主要专家始终是 Expert 0。

### 3. State-Expert alignment 不支持强状态专门化

状态-专家热力图中，各个 state component 到 expert 的平均 router probability 非常接近。

这表明 Student-T state prior 的不同状态没有被 router 映射到明显不同的专家。

因此，目前不能说“不同状态组件对应不同专家”。

### 4. Feature-by-expert 图可以用于观察次级专家偏好

由于 Top-1 塌缩，feature 图改为按 Top-K membership 分组，而不是按 Top-1 分组。

该图用于观察：被某个专家选入 Top-K 的样本，是否在输入波动、未来变化、路由熵等方面有明显差异。

从当前结果看，Abilene 中不同次级专家存在一些分布差异，但 Geant 主要只在 Expert 0 和 Expert 3 之间形成 Top-K 搭配。

## 结论

在当前 Abilene / Geant 的 PL5 round0 checkpoint 上，实验结果不支持“Top-K MoE 已形成强专家专门化”这个结论。

更准确的说法是：

```text
当前模型存在 Top-K 次级专家参与，但 Top-1 路由明显塌缩到 Expert 0；
state-expert alignment 较弱，专家专门化证据不足。
```

如果论文中要强调专家专门化，建议进一步加入以下约束或对照实验：

- 提高 MoE load-balancing loss 权重。
- 对 Top-1 usage 加入均衡约束。
- 降低 router softmax 温度，让路由更尖锐。
- 对专家输出加入 diversity regularization。
- 比较无 load-balancing、弱 load-balancing、强 load-balancing 三种设置下的专家使用图。
- 在多个 round 和多个 pred_len 上重复该分析，避免单个 checkpoint 偶然性。

## 复现实验

运行命令：

```bash
python MoEExpertAnalysis/run_moe_specialization_analysis.py --datasets Abilene Geant --pred_lens 5 --round_id 0 --device cpu
```

脚本位置：

```text
MoEExpertAnalysis/run_moe_specialization_analysis.py
```
