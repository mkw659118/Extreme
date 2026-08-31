# 两阶段线性插值预测实验

## 实验目标

比较在相同历史输入缺失条件下，不同预测 baseline 的两阶段流程：

```text
历史输入缺失 -> 线性插值 -> 预测模型 -> 完整未来标签
```

本实验不进行张量补全，也不修改未来标签。它验证的是线性插值预处理后的独立预测模型表现。

## 两种缺失配置

| 配置 | 缺失模式 | 缺失率 | 块长度 | 变量覆盖 | 数据范围 | 种子 |
|---|---|---:|---:|---:|---|---:|
| Random Missing Rate = 5% | `random_point` | 5% | - | 全部变量 | train/val/test | 2026 |
| Structured Time-Block Missing Rate = 20% | `time_block` | 20% | 12 | 100% | train/val/test | 2026 |

人工缺失只改变历史输入的可见性。训练、验证和测试的未来标签均使用原始观测掩码，不施加人工缺失。

## 插值规则

插值在每个长度为 `seq_len` 的历史窗口内部独立执行，因此不会读取该窗口之后的未来标签：

- 内部缺口：相邻观测值之间进行线性插值；
- 窗口左边界缺失：使用窗口内第一个观测值向前填充；
- 窗口右边界缺失：使用窗口内最后一个观测值向后填充；
- 某变量在整个窗口内都缺失：使用训练集该变量均值；
- 插值完成后，再使用训练集统计量进行标准化。

DATP-Net直接缺失实验仍使用“数值占位 + 显式mask”，不应对DATP-Net执行线性插值。

## Baseline

默认运行11个现有预测模型：PMDformer、HMformer、FeTS、TimesNet、iTransformer、FEDformer、PatchTST、WPMixer、P-sLSTM、xLSTMTime、xLSTM-Mixer。

两个数据集为Abilene和GÉANT。默认预测长度为5，`d_model=256`，每项配置重复5次。默认总训练数为：

```text
2个缺失配置 × 2个数据集 × 11个模型 × 5次重复 = 220次训练
```

## 运行

在项目根目录执行：

```bash
bash script/baseline_linear_imputation_table.sh
```

如果当前系统的默认 `python` 不是项目训练环境，可指定解释器：

```bash
PYTHON_BIN=/path/to/conda/env/bin/python bash script/baseline_linear_imputation_table.sh
```

先进行快速验证：

```bash
ROUNDS=1 EPOCHS=2 PATIENCE=1 MODELS="PMDformerConfig PatchTSTConfig" \
  bash script/baseline_linear_imputation_table.sh
```

如果需要多个预测长度：

```bash
PRED_LENS="5 10 15 20" bash script/baseline_linear_imputation_table.sh
```

## 结果文件

每次训练的原始指标保存在 `results/metrics/`。全部配置完成后，脚本自动在 `results/two_stage_linear/` 生成：

- CSV：便于Excel继续排版；
- Markdown：便于直接检查；
- LaTeX：可直接放入论文表格。

表格按以下两个分区组织：

1. `Random Missing Rate = 5%`；
2. `Structured Time-Block Missing Rate = 20%`。

每个模型显示Abilene和GÉANT上的NMAE、NRMSE、COS，单元格格式为“均值 ± 标准差”。NMAE和NRMSE越小越好，COS越大越好。
