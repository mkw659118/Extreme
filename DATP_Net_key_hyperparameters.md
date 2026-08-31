# DATP-Net 关键超参数与实现细节补充

本文档用于回应“关键超参数和实现细节未给出”的问题。以下数值均按照当前 DATP-Net 代码实现和实验脚本整理。

## 1. Mask-Aware 相似度

对于当前查询窗口和第 \(i\) 个历史 key，DATP-Net 只在双方共同有效的位置上计算相似度。令展平后的查询值为 \(\mathbf q_t\)，历史 key 值为 \(\mathbf k_i\)，对应有效掩码为 \(\mathbf m_t\) 和 \(\mathbf m_i\)，则 mask-aware 相似度为

\[
\bar{s}_{t,i} =
\frac{
\sum_j q_{t,j} k_{i,j} m_{t,j}m_{i,j}
}{
\sqrt{\sum_j q_{t,j}^2 m_{t,j}m_{i,j}}
\sqrt{\sum_j k_{i,j}^2 m_{t,j}m_{i,j}}+\epsilon
}.
\]

具体数值：

- \(\epsilon = 10^{-8}\)
- 有效重叠阈值：至少存在 1 个共同有效位置
- 等价写法：\(\sum_j m_{t,j}m_{i,j} > 0\)
- 若无共同有效位置，则相似度置为 \(-1.0\)
- 训练阶段排除时间重叠窗口：排除范围为 \([t-L,t+L]\)
- 输入窗口长度：\(L=96\)

## 2. 历史记忆库大小

历史记忆库记为 \(\mathcal B\)。它不是固定常数，而是由训练集窗口数决定：

\[
|\mathcal B| = N_{\mathrm{train}}.
\]

实现中每个训练窗口写入一组 key-value：

- key shape: \([|\mathcal B|, L, C]\)
- value shape: \([|\mathcal B|, H, C_y]\)
- 其中 \(L=96\)，\(H=\text{pred\_len}\)

论文中可以写为：

> The memory size is set to the number of training windows, i.e., \(|\mathcal B|=N_{\mathrm{train}}\), with one key-value pair stored for each training sample.

## 3. 检索与融合参数

DATP-Net 每次检索最相似的 \(m\) 个历史样本：

- retrieved cases: \(m=2\)
- 检索权重 softmax 温度：\(\tau=1.0\)
- 检索启发式阈值：\(\tau_r=0.55\)
- 启发式融合上界：\(\alpha_{\max}=0.02\)
- gate 融合上界：\(\lambda_{\max}=0.1\)
- gate 正则权重：\(10^{-4}\)

代码中的最终 gate 融合形式为：

\[
\hat{Y}_t = (1-\lambda_t)\hat{Y}^{moe}_t + \lambda_t \hat{Y}^{his}_t,
\quad 0 \le \lambda_t \le 0.1.
\]

因此，论文中建议将 \(\lambda_{\max}\) 写为：

\[
\lambda_{\max}=0.1.
\]

## 4. MoE 与状态先验参数

DATP-Net 使用 Student-t 状态先验引导 Top-K MoE 路由。

具体数值：

- latent state 数：\(C=4\)
- expert 数：\(K=4\)
- 激活 expert 数：\(K_r=2\)
- 多尺度集合：\(\mathcal S=\{1,4,8,16\}\)
- 是否加入 sequence-level descriptor：是
- Student-t posterior 温度：\(1.0\)
- 预训练温度调度：\(1.0 \rightarrow 0.6\)

## 5. 正则项权重

当前代码中的主要正则权重如下。

| 符号 | 建议对应代码项 | 具体数值 |
|---|---|---:|
| \(\lambda_s\) | Student-t component diversity weight | \(0.001\) |
| \(\lambda_b\) | sample-level router balance weight | \(0.1\) |
| \(\lambda_{\max}\) | reliability-bounded fusion upper bound | \(0.1\) |
| \(\lambda_{\text{trend}}\) | trend-consistency loss weight | 当前代码未实现同名项 |

补充说明：

- state balance weight: \(0.02\)
- state dominance cap: \(0.8\)
- state assignment entropy weight: \(0.0005\)
- Student-t mean separation margin: \(1.0\)
- Student-t scale separation margin: \(0.3\)
- Student-t degree-of-freedom separation margin: \(0.2\)
- scale diversity internal weight: \(0.2\)
- df diversity internal weight: \(0.1\)
- Top-K coverage weight: \(0.25\)
- Top-K minimum usage: \(0.12\)
- MoE max weight: \(0.55\)
- MoE min weight: \(0.08\)

## 6. 训练参数

实验中的主要训练参数为：

- batch size: \(B=32\)
- learning rate: \(5\times 10^{-4}\)
- optimizer: Adam
- maximum epochs: 200
- patience: 40
- sequence length: 96
- prediction horizons: 5, 10, 15, 20
- default prediction horizon in main short-horizon experiments: 5
- hidden dimension in main DATP-Net experiments: 256

## 7. 可直接加入论文的英文段落

```latex
For mask-aware retrieval, the similarity between the current query and
the $i$-th historical key is computed only on jointly valid positions:
\[
\bar{s}_{t,i} =
\frac{
\sum_j q_{t,j} k_{i,j} m_{t,j}m_{i,j}
}{
\sqrt{\sum_j q_{t,j}^2 m_{t,j}m_{i,j}}
\sqrt{\sum_j k_{i,j}^2 m_{t,j}m_{i,j}}+\epsilon
},
\]
where $\epsilon=10^{-8}$. A candidate is admissible if it has at least
one jointly observed position with the query; otherwise its similarity is
set to $-1$. During training, temporally overlapping windows within one
input length are excluded to avoid information leakage.

Unless otherwise specified, DATP-Net uses batch size $B=32$, $K=4$
experts, $K_r=2$ activated experts, $m=2$ retrieved historical cases,
and multi-scale state-prior scales $\mathcal S=\{1,4,8,16\}$ with an
additional sequence-level descriptor. The memory size is
$|\mathcal B|=N_{\mathrm{train}}$, i.e., one key-value entry is stored
for each training window. The retrieval softmax temperature is set to
$\tau=1.0$, and the Student-$t$ posterior temperature is initialized as
$1.0$ and annealed to $0.6$ during state-prior pretraining. The
reliability-bounded fusion weight is capped by $\lambda_{\max}=0.1$.
The state diversity and router balance weights are set to
$\lambda_s=0.001$ and $\lambda_b=0.1$, respectively.
```

## 8. 关于 \(\lambda_{\text{trend}}\) 的处理建议

当前文章中出现了 \(\lambda_{\text{trend}}L_{\text{trend}}\)，但当前 DATP-Net 主实现中没有找到同名 trend-consistency loss 权重。因此有两个处理方式：

1. 若保留 trend loss，需要在代码和实验设置中明确加入该项，并给出固定值。
2. 若按当前代码实事求是修改，建议删除 \(\lambda_{\text{trend}}\) 相关描述，或改写为“trend preservation is evaluated by COS rather than optimized by an explicit trend loss”。

如果必须给出一个表中数值，建议不要凭空填写 \(\lambda_{\text{trend}}\)，应标注为 “not used in the final implementation”。
