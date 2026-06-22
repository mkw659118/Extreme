# 容量预留下游任务说明文档

本文档说明以下两张容量预留下游任务图：

- `abilene_all_provisioning_tradeoff.pdf`
- `geant_all_provisioning_tradeoff.pdf`

这两张图对应 Abilene 和 G&Eacute;ANT 两个数据集上的测试集整体结果。这里的 `all` 表示使用测试集中的全部时间点进行统计。

## 1. 实验目的

这个下游任务的目标是评估预测模型在网络运维中的实际价值。传统预测指标，例如 MAE、RMSE 或 NMAE，主要衡量预测值和真实值之间的平均误差；但在网络容量管理中，更关键的问题通常是：

- 预测结果能不能帮助提前预留足够容量？
- 在控制 SLA 违约率的同时，是否能减少过度预留带来的带宽浪费？
- 模型是否会因为过度平滑而低估突发流量，从而导致预留不足？

因此，本实验把流量预测结果转化为容量预留决策。一个好的预测模型不仅应该平均误差低，还应该能够在同样 SLA 违约目标下使用更少的额外容量。

## 2. 容量预留规则

对于每个测试时间点 `t`，模型会给出一个流量预测值：

```text
y_hat_t
```

根据预测值和安全裕度 `alpha`，预留容量定义为：

```text
C_t = y_hat_t * (1 + alpha)
```

其中：

- `y_hat_t` 是模型预测的流量需求。
- `alpha` 是安全裕度，也可以理解为 over-provisioning margin。
- `C_t` 是根据预测结果实际预留的容量。
- `y_t` 是该时间点的真实流量需求。

例如，当 `alpha = 0.2` 时，表示在预测值基础上额外预留 20% 的容量：

```text
C_t = 1.2 * y_hat_t
```

如果模型预测偏低，尤其是在突发流量或快速上升趋势中低估真实需求，那么即使加上安全裕度，也可能出现容量不足。

## 3. SLA 违约如何计算

如果真实流量需求超过预留容量，则认为该时间点发生 SLA 违约：

```text
y_t > C_t
```

SLA 违约率定义为：

```text
SLA violation rate
= count(y_t > C_t) / number_of_test_slots
```

也就是：

```text
违约时间点数量 / 测试时间点总数
```

这个指标越低，说明模型驱动的容量预留越可靠，越不容易出现 under-provisioning。

## 4. 过度预留成本如何计算

如果预留容量超过真实需求，多出来的部分可以理解为被浪费的带宽资源：

```text
max(0, C_t - y_t)
```

所有测试时间点上的总过度预留成本为：

```text
sum(max(0, C_t - y_t))
```

为了让不同数据集之间更容易比较，图中使用的是归一化过度预留成本：

```text
normalized over-provisioning cost
= sum(max(0, C_t - y_t)) / sum(y_t)
```

也就是：

```text
总浪费容量 / 总真实流量需求
```

这个值越低，说明为了保证 SLA 所付出的额外容量成本越低。

## 5. 图中横轴和纵轴含义

两张图的横轴是：

```text
Normalized over-provisioning cost
```

含义是归一化过度预留成本。横轴越靠左，说明带宽浪费越少。

两张图的纵轴是：

```text
SLA violation rate
```

含义是 SLA 违约率。纵轴越靠下，说明容量不足的情况越少。

因此，这类图的核心判断标准是：

```text
曲线越靠左下角越好。
```

左下角代表：

- 过度预留成本低；
- SLA 违约率也低；
- 即用更少的额外容量实现更可靠的服务保障。

## 6. 曲线是怎么得到的

每条曲线对应一个预测模型。实验中对安全裕度 `alpha` 进行扫描：

```text
alpha = 0.00, 0.01, 0.02, ..., 2.00
```

对于每一个 `alpha`，都计算一组容量预留结果：

```text
C_t = y_hat_t * (1 + alpha)
```

然后统计该 `alpha` 下的：

- SLA violation rate；
- normalized over-provisioning cost。

把不同 `alpha` 对应的点连起来，就得到图中的一条 trade-off 曲线。

随着 `alpha` 增大，模型预留的容量会变多，因此通常会出现：

- SLA 违约率下降；
- 过度预留成本上升。

也就是说，沿着一条曲线从左上走向右下，本质上是在用更多额外容量换取更低违约率。

## 7. 如何比较两个模型

比较两个模型时，不能只看某一个点，而应该看整条曲线的位置。

如果模型 A 的曲线整体比模型 B 更靠左下角，说明：

```text
在相同 SLA 违约率下，模型 A 需要更少的过度预留成本；
或者在相同过度预留成本下，模型 A 能达到更低的 SLA 违约率。
```

这就是容量预留下游任务的核心意义：它不只是比较预测误差，而是比较预测结果转化成运维决策之后的实际代价。

## 8. DATP-Net 在图中的标识

在两张图中，DATP-Net 使用橙色粗线并带有圆点标记。其他 baseline 使用较细的普通曲线。

这样设计是为了突出本文方法和其他模型之间的下游 trade-off 差异。

## 9. DATP-Net 的优势在哪里

从 Abilene 和 G&Eacute;ANT 两张图可以看到，DATP-Net 的橙色曲线整体位于更靠左下的位置。这说明 DATP-Net 在容量预留下游任务中具有更好的 SLA-cost trade-off。

具体来说，DATP-Net 的优势体现在两个方面。

第一，在相同 SLA 违约目标下，DATP-Net 需要的过度预留成本更低。例如，当要求 SLA violation rate 不超过某个目标值时，DATP-Net 通常可以用更小的 normalized over-provisioning cost 达到该目标。这意味着在达到相同可靠性要求的情况下，DATP-Net 可以减少额外带宽预留，降低资源浪费。

第二，在相同过度预留成本下，DATP-Net 往往能取得更低的 SLA 违约率。也就是说，如果运维系统只能接受一定程度的额外容量开销，DATP-Net 可以更有效地把这些容量用在需要的地方，从而减少 under-provisioning。

这种优势说明 DATP-Net 并不是简单地把预测值整体抬高来减少违约。如果一个模型只是系统性高估流量，它也可以降低 SLA 违约率，但会导致横轴上的 over-provisioning cost 很大。DATP-Net 的优势在于，它能够在较低过度预留成本下取得较低违约率，因此曲线更靠近左下角。

## 10. 为什么这个结果能体现趋势保持的价值

容量预留任务尤其关注模型是否会低估高峰和上升趋势。对于过度平滑的模型，即使它的平均误差不一定很差，也可能出现以下问题：

```text
突发或上升趋势被抹平
-> 预测值偏低
-> 预留容量不足
-> SLA 违约增加
```

为了弥补这种低估，过度平滑模型只能依赖更大的 `alpha`，也就是整体提高安全裕度。然而这样会带来更高的过度预留成本。

DATP-Net 的曲线更靠左下，说明它在趋势变化和峰值区域的预测更有运维价值：它不需要依靠非常大的全局安全裕度，也能更好地控制 SLA 违约。这与 DATP-Net 强调的 trend-preserving 能力是一致的。

## 11. 可以在论文中如何表述

这两张图可以支撑如下结论：

```text
DATP-Net achieves a better capacity provisioning trade-off than strong forecasting baselines. Under the same SLA violation target, DATP-Net requires less over-provisioned capacity; under the same provisioning cost, it leads to fewer SLA violations.
```

对应中文表述为：

```text
DATP-Net 在容量预留下游任务中取得了更优的 SLA-cost 权衡。在相同 SLA 违约率目标下，DATP-Net 所需的过度预留容量更少；在相同容量开销下，DATP-Net 带来的 SLA 违约率更低。
```

如果要强调运维意义，可以写成：

```text
These results show that DATP-Net's forecasting improvements translate into practical operational benefits, reducing bandwidth waste while maintaining SLA reliability.
```

对应中文表述为：

```text
这些结果表明，DATP-Net 的预测优势能够转化为实际运维收益，即在维持 SLA 可靠性的同时减少带宽浪费。
```

## 12. 总结

`abilene_all_provisioning_tradeoff.pdf` 和 `geant_all_provisioning_tradeoff.pdf` 展示的是预测模型驱动容量预留时的成本和可靠性权衡。

读图时需要关注：

- 横轴越小越好，表示过度预留成本越低；
- 纵轴越小越好，表示 SLA 违约率越低；
- 曲线越靠左下角越好；
- DATP-Net 的橙色曲线整体更靠左下，说明它在相同 SLA 目标下更省容量，在相同容量成本下更少违约。

因此，这两张图可以作为下游任务证据，说明 DATP-Net 的预测结果不仅在误差指标上有效，而且能带来更好的容量预留决策价值。

