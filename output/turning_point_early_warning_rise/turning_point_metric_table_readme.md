# Turning-point Early-warning 表格说明

本文档说明 `turning_point_metric_table.tex` 中的数值型实验结果。该表对应主动扩容及时性实验，也就是 turning-point early-warning downstream task。

## 1. 实验目的

这个实验的目标不是单纯比较预测误差，而是评估预测模型能否帮助网络运维系统更早发现即将到来的高负载风险。

在实际网络运维中，如果流量即将进入高负载或接近容量风险区，系统希望能够提前触发扩容、资源调度或告警。一个预测模型如果过度平滑，可能会把快速上升趋势或突发峰值抹平，导致预警过晚甚至漏报。相反，一个更能保持趋势和峰值的模型，应该能够在真实流量越过风险阈值之前提前给出预警。

因此，本实验关注三个问题：

- 模型能不能在真实越界前触发预警？
- 如果能够预警，能提前多少个时间片？
- 模型是否为了提高命中率而产生大量误报？

## 2. 使用的数据

该实验使用每个模型在测试集上的多步预测结果 `test_raw.csv`。

当前预测长度为：

```text
PredLen = 5
```

也就是说，在每个测试窗口起点，模型会预测未来 5 个时间点的流量：

```text
pred_{t+1}, pred_{t+2}, pred_{t+3}, pred_{t+4}, pred_{t+5}
```

实验中包含两个数据集：

- Abilene
- G&Eacute;ANT

其中 Abilene 当前使用的是数据集尺度下的流量值，不是 Mbps 或 Gbps 等物理单位。因此本文中的阈值和流量值都应理解为 dataset scale 下的数值。

## 3. 风险分数如何计算

为了把多步预测转化为主动扩容预警信号，实验对每个时间点计算一个风险预测值：

```text
risk_pred_t = max(pred_{t+1}, pred_{t+2}, pred_{t+3}, pred_{t+4}, pred_{t+5})
```

也就是说，模型只要认为未来 5 个时间点中任意一个时间点可能达到高负载，就会把当前窗口视为存在扩容风险。

这样做的原因是，主动扩容关注的是“未来一小段时间内是否会出现风险”，而不是只关心下一步预测是否准确。

## 4. 扩容阈值如何定义

由于当前实验没有使用真实链路容量，因此扩容阈值使用测试集真实流量的 90% 分位数定义：

```text
threshold = quantile(true_test, 0.90)
```

当真实流量超过该阈值时，认为系统进入高负载风险状态：

```text
true_t > threshold
```

当模型的风险预测值超过该阈值时，认为模型触发主动扩容预警：

```text
risk_pred_t > threshold
```

## 5. Turning-point 事件如何选择

本表使用的是中立事件选择方式：

```text
selection_mode = rise
```

这意味着事件选择只依赖真实流量本身，而不是根据 DATP-Net 的表现挑选案例。

具体来说，实验寻找真实流量从阈值以下越过阈值的急升点：

```text
true_{t-1} <= threshold
true_t > threshold
```

然后按照真实流量上升幅度选择代表性的 turning-point 事件。

当前每个数据集选择 3 个事件：

```text
Abilene: 3 cases
G&Eacute;ANT: 3 cases
```

由于事件选择不依赖模型预测结果，因此该表比手工挑选 case study 图更适合作为数值型主结果或补充结果。

## 6. 预警提前量如何计算

对于每一个真实越界事件，设真实越界时间为：

```text
t_cross
```

对于某个模型，如果它在 `t_cross` 之前或刚好在 `t_cross` 时首次触发预警，设预警时间为：

```text
t_alarm
```

则提前量定义为：

```text
lead_time = t_cross - t_alarm
```

如果：

```text
lead_time > 0
```

表示模型提前预警。

如果：

```text
lead_time = 0
```

表示模型在真实越界时刻同步预警。

如果模型在真实越界前没有触发预警，则记为 missed alarm。

## 7. 表格列含义

表格中的列含义如下。

### Dataset

数据集名称，包括 Abilene 和 G&Eacute;ANT。

### Model

预测模型名称。

### Early

成功提前预警的事件数量，格式为：

```text
成功预警数 / 总事件数
```

例如：

```text
2/3
```

表示 3 个真实 turning-point 事件中，有 2 个事件模型在真实越界前或越界时成功触发预警。

该指标越高越好。

### Miss

漏报次数。表示真实流量发生越界事件，但模型没有在事件前或事件时触发预警。

该指标越低越好。

### Delay

滞后预警次数。表示模型没有提前预警，而是在真实越界之后才触发预警。

该指标越低越好。

当前表中 Delay 均为 0，说明在设定的搜索窗口内，没有模型出现被统计为 delayed alarm 的情况；未能提前触发的情况主要表现为 missed alarm。

### Mean Lead

成功提前预警事件上的平均提前量，单位是 time slots。

例如：

```text
Mean Lead = 28.0
```

表示在成功预警的事件中，模型平均提前 28 个时间片触发扩容预警。

该指标越高，说明预警越早。

### Median Lead

成功提前预警事件上的中位数提前量，单位同样是 time slots。

它比平均值更不容易受个别极端事件影响。

### FAR (%)

False Alarm Rate，误报率。这里的 FAR 定义为：

```text
FAR = false_alarm_windows / alarm_windows
```

其中：

```text
alarm_windows
```

表示模型预测未来 5 步内会越过阈值的窗口数量。

```text
false_alarm_windows
```

表示模型发出预警，但真实未来 5 步并没有越过阈值的窗口数量。

因此 FAR 衡量的是：

```text
模型触发的预警中，有多少比例是误报。
```

该指标越低越好。

## 8. 如何读这个表

读表时不要只看 `Early`，也不要只看 `FAR`，而应该同时看三类指标：

```text
Early / Miss / Delay: 是否能及时发现真实风险
Mean Lead / Median Lead: 能提前多久发现风险
FAR: 是否通过大量误报换取命中率
```

一个理想模型应该满足：

- Early 高；
- Miss 低；
- Delay 低；
- Mean Lead 和 Median Lead 较高；
- FAR 较低。

如果一个模型 Early 很高，但 FAR 也很高，说明它可能是通过频繁报警来覆盖真实事件。这种模型在实际运维中会带来告警疲劳和不必要的扩容成本。

如果一个模型 FAR 很低，但 Early 也很低，说明它过于保守，容易漏掉真实风险。

因此，该表重点比较的是预警及时性和误报控制之间的平衡。

## 9. DATP-Net 的结果如何解读

### Abilene

DATP-Net 在 Abilene 上的结果为：

```text
Early = 1/3
Miss = 2
Delay = 0
Mean Lead = 30.0
Median Lead = 30.0
FAR = 27.69%
```

这说明在中立选择的 3 个真实急升事件中，DATP-Net 成功提前预警了 1 个事件，漏报了 2 个事件。在成功预警的事件上，DATP-Net 提前 30 个时间片触发预警。

在 Abilene 上，多个 baseline 也表现为 `Early = 1/3`，因此 DATP-Net 在命中率上不是唯一最优。但 DATP-Net 的 FAR 为 27.69%，低于许多误报较高的模型，例如 TimesNet、iTransformer、PatchTST、WPMixer、P-sLSTM、xLSTMTime 和 xLSTM-Mixer。

因此，Abilene 上更合适的结论是：

```text
DATP-Net achieves competitive early-warning performance with a relatively low false alarm rate.
```

也就是，DATP-Net 的预警命中率与多数模型相当，同时误报率处于较低水平。

### G&Eacute;ANT

DATP-Net 在 G&Eacute;ANT 上的结果为：

```text
Early = 2/3
Miss = 1
Delay = 0
Mean Lead = 28.0
Median Lead = 28.0
FAR = 21.15%
```

这说明在 3 个真实急升事件中，DATP-Net 成功提前预警了 2 个事件，平均提前 28 个时间片。

虽然部分 baseline 达到了 `Early = 3/3`，但它们的 FAR 明显更高。例如：

```text
PMDformer: FAR = 52.63%
FeTS: FAR = 54.12%
TimesNet: FAR = 54.05%
iTransformer: FAR = 43.02%
xLSTM-Mixer: FAR = 45.76%
FEDformer: FAR = 45.31%
```

相比之下，DATP-Net 的 FAR 为：

```text
21.15%
```

这是 G&Eacute;ANT 上最低的误报率。

因此，G&Eacute;ANT 上可以得出较明确的结论：

```text
DATP-Net provides a more reliable early-warning signal by maintaining competitive lead time while substantially reducing false alarms.
```

也就是说，DATP-Net 不是通过频繁报警来提高命中率，而是在保持较好提前量的同时显著减少误报。

## 10. 可以写进论文的结论

这张表不应该被解释为 DATP-Net 在所有 early-warning 指标上全面最优。更准确的结论是：

```text
Under neutral rise-based event selection, DATP-Net achieves a favorable trade-off between early warning and false alarm control. In particular, on G&Eacute;ANT, DATP-Net obtains the lowest false alarm rate while maintaining competitive lead time.
```

中文表述为：

```text
在中立的急升事件选择下，DATP-Net 在提前预警和误报控制之间取得了较好的平衡。尤其在 G&Eacute;ANT 上，DATP-Net 在保持有竞争力提前量的同时取得了最低误报率。
```

## 11. 总结

该表格展示的是主动扩容任务中的数值型结果。相比单个 case study 图，它更适合说明模型在多个真实 turning-point 事件上的整体表现。

主要结论是：

- DATP-Net 在 Abilene 上具有有竞争力的提前预警能力和较低误报率；
- DATP-Net 在 G&Eacute;ANT 上误报率最低，同时提前量保持在较高水平；
- DATP-Net 的优势更适合表述为“稳健预警”和“误报控制”，而不是“所有指标全面最优”。

