# DATP-Net 结构化时间块缺失消融实验说明

## 1. 实验目的

本实验验证 DATP-Net 的四个组件在结构化时间块缺失条件下是否分别贡献有效性能：

- Mixture of Experts（MoE）；
- Retrieval；
- State Prior；
- Missing-Aware Encoding。

所有模型在相同数据划分、相同人工缺失掩码、相同训练超参数和相同随机种子集合下从头训练。每次只关闭一个组件。

## 2. 缺失机制

人工缺失只施加到模型输入，不改变预测标签或标签有效性掩码。数据加载器先从原始数据生成真实观测掩码 `full_mask`，再从该掩码复制出 `input_mask` 并施加人工缺失：

```text
模型输入：input_mask（真实缺失 + 人工时间块缺失）
监督标签：full_mask（仅保留数据本身的真实缺失）
```

`time_block` 模式生成时间 × 变量的连续矩形缺失块。主要参数为：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `artificial_missing_rate` | 0.20 | 每个数据划分中目标人工缺失比例 |
| `artificial_missing_block_length` | 12 | 每个缺失块连续覆盖的时间步数 |
| `artificial_missing_column_rate` | 1.0 | 每个块覆盖的变量比例；1.0 表示所有变量 |
| `artificial_missing_seed` | 2026 | 缺失掩码随机种子 |
| `artificial_missing_splits` | train,val,test | 应用人工缺失的数据划分 |

算法会不断采样连续块，直到每个选定数据划分达到目标删除数量。由于最后一个完整块可能越过目标数量，实际缺失率可能略高于目标值；日志中的 `[Block Missing] actual_rate=...` 是应在论文中记录的实际比例。

相同数据集和缺失配置使用同一个 `artificial_missing_seed`，因此完整模型与四个消融模型获得完全一致的缺失掩码。

## 3. 消融定义

| 变体 | Missing-Aware | State Prior | Retrieval | Experts / Top-K |
|---|---:|---:|---:|---:|
| `full` | 开启 | 开启 | 开启 | 4 / 2 |
| `wo_moe` | 开启 | 开启 | 开启 | 1 / 1 |
| `wo_retrieval` | 开启 | 开启 | 关闭 | 4 / 2 |
| `wo_state_prior` | 开启 | 关闭 | 开启 | 4 / 2 |
| `wo_missing_aware_encoding` | 关闭 | 开启 | 开启 | 4 / 2 |

### w/o MoE

设置 `num_experts=1`、`top_k_experts=1`，使 MoE 退化为单一专家。State Prior 仍保留四个状态，但不再产生多专家选择收益。

### w/o Retrieval

设置 `use_retrieval=False`。训练流程跳过检索索引构建，模型前向过程不执行检索预测融合。

### w/o State Prior

设置 `use_state_prior=False`。Student-t State Prior 被关闭，MoE 改由基于编码特征的 learned router 产生路由权重。因此该变体准确含义是“以普通 learned router 替代 State Prior”。

### w/o Missing-Aware Encoding

设置 `use_missing_aware_encoding=False`。输入仍使用同一个人工缺失掩码和零占位值，但不再向模型提供 `diff_mask`、`second_diff_mask` 和 `raw_mask` 三组显式缺失标记。

完整输入特征为：

```text
[diff_norm, diff_mask, second_diff_raw, second_diff_mask, raw, raw_mask]
```

消融输入特征为：

```text
[diff_norm, second_diff_raw, raw]
```

## 4. 运行方法

在项目根目录执行：

```bash
bash script/DATPNet_block_missing_ablation.sh
```

脚本通过独立的 DATP-Net 专用入口 `run_train_DATPNet.py` 启动。该文件
自行实现配置解析、实验命名、多轮训练、指标汇总和日志流程，不导入或依赖
`run_train_DARNet.py`，因此以后删除旧 DARNet 启动文件不会影响本实验。
模型构造使用 DATP-Net 专用适配器 `exp/exp_model_DATPNet.py`，该适配器
只导入 `modules/DATP_Net.py`，不会加载旧的 `modules/DARNet1.py`。
入口默认读取 `DATPNetConfig`，并检查 `config.model` 必须是 `datp_net`、
`datp_net_step` 或 `datp_net_horizon`，从入口层防止误跑旧版 DARNet。

默认运行：

- 数据集：Abilene、Geant；
- 预测长度：5、10、15、20；
- 五个模型变体；
- 每个配置 5 次重复；
- 20% 时间块缺失；
- 块长度 12；
- 每个块覆盖所有变量。

完整默认实验共运行 `2 × 4 × 5 = 40` 个配置，每个配置内部进行 5 次重复训练。

### 快速冒烟测试

```bash
DATASETS="Geant" \
PRED_LENS="5" \
D_MODEL=16 \
EPOCHS=1 \
PATIENCE=1 \
ROUNDS=1 \
BATCH_SIZE=64 \
PRETRAIN_EPOCHS=0 \
GATE_EPOCHS=0 \
bash script/DATPNet_block_missing_ablation.sh
```

### 参数覆盖示例

运行 40% 缺失、连续 20 步、每个块遮蔽 30% 变量：

```bash
MISSING_RATE=0.40 \
BLOCK_LENGTH=20 \
COLUMN_RATE=0.30 \
bash script/DATPNet_block_missing_ablation.sh
```

脚本支持以下环境变量：

| 环境变量 | 默认值 |
|---|---|
| `PYTHON_BIN` | `python` |
| `CONFIG` | `DATPNetConfig` |
| `DATASETS` | `Abilene Geant` |
| `PRED_LENS` | `5 10 15 20` |
| `D_MODEL` | `256` |
| `SEQ_LEN` | `96` |
| `EPOCHS` | `200` |
| `PATIENCE` | `40` |
| `ROUNDS` | `5` |
| `BATCH_SIZE` | `32` |
| `SEED` | `2026` |
| `PRETRAIN_EPOCHS` | `20` |
| `GATE_EPOCHS` | `5` |
| `MISSING_RATE` | `0.20` |
| `BLOCK_LENGTH` | `12` |
| `COLUMN_RATE` | `1.0` |
| `MISSING_SPLITS` | `train,val,test` |

## 5. 正式运行前检查

每个任务启动后检查日志：

1. `Model : datp_net`，确保运行的是 DATP-Net 而不是旧版 `net`；
2. 出现 `[Block Missing]`，确保不是随机点缺失；
3. 五个变体的 `actual_rate` 和每个 split 的删除数量一致；
4. 实验详情中记录 `Artificial_Missing_Pattern : time_block`；
5. `wo_retrieval` 日志出现跳过检索索引的提示；
6. 每个变体都创建独立 checkpoint，且 `retrain=True`。

## 6. 结果报告

建议将结果填入以下表格，并报告多次运行的均值和标准差：

| Model | NMAE ↓ | NRMSE ↓ | COS ↑ | MLU ratio ↓ | 参数量 |
|---|---:|---:|---:|---:|---:|
| DATP-Net |  |  |  |  |  |
| w/o MoE |  |  |  |  |  |
| w/o Retrieval |  |  |  |  |  |
| w/o State Prior |  |  |  |  |  |
| w/o Missing-Aware Encoding |  |  |  |  |  |

对于越小越好的误差指标，可报告相对退化率：

```text
Degradation = (Metric_ablation - Metric_full) / Metric_full × 100%
```

由于 `w/o MoE` 和 `w/o Missing-Aware Encoding` 会改变模型参数量，结果表中应同时报告参数量，避免把容量差异误解为纯组件贡献。

## 7. 可复现性说明

- 数据划分仍使用原项目的顺序 70%/10%/20% 划分；
- 缺失块不会跨越 train、validation 和 test 的边界；
- 验证集和测试集仍保留前 `seq_len` 个历史点作为输入上下文；
- 所有消融共享相同缺失种子；
- `rounds` 改变模型初始化种子，但默认不改变缺失掩码，因此测量的是固定缺失场景下的训练随机性；
- 若需要同时评估不同缺失实现，应分别改变 `SEED` 并让五个消融模型继续共享每个对应 seed。

## 8. 实现验证记录

本次交付完成了以下可用性测试：

1. Python 语法编译检查通过；
2. Git 空白符检查通过；
3. 固定 seed 重复构造数据集，两次 `input_mask` 完全相同；
4. Abilene、20% 缺失、块长度 12 的测试中，实际缺失率为 20.57%，最短连续缺失段为 12；
5. 完整模型与 `w/o Missing-Aware Encoding` 获得相同缺失掩码，输入维度分别为 6 和 3；
6. 五个变体均完成一个真实 batch 的 DATP-Net 前向传播，输出形状均为 `[batch, pred_len, 1]`，且输出无 NaN/Inf；
7. 完整 DATP-Net 完成一轮端到端冒烟训练，包括训练、验证、检索索引构建、测试、指标计算和 checkpoint 保存；
8. 冒烟运行日志确认 `Model : datp_net` 且记录了 `Artificial_Missing_Pattern : time_block`。

运行器原先会在配置加载和 Logger 初始化时扫描所有历史结果并删除无效日志。新脚本显式使用 `--skip_startup_cleanup True`，避免 40 组正式实验重复执行这些旧维护操作；普通旧脚本保持原行为不变。
