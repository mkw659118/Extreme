

# DATP-Net 关键超参数



## Mask-aware 相似度

DATP-Net 在检索历史窗口时，只使用当前窗口和历史 key 都有效的位置计算相似度。

公式可以写为：

```latex
s_bar(t,i) =
sum_j q(t,j) * k(i,j) * m(t,j) * m(i,j)
/
(
sqrt(sum_j q(t,j)^2 * m(t,j) * m(i,j))
*
sqrt(sum_j k(i,j)^2 * m(t,j) * m(i,j))
+ epsilon
)
```


具体数值：

- epsilon = 1e-8
- **有效重叠阈值：至少 1 个共同有效位置**
- 等价条件：sum_j m(t,j) * m(i,j) > 0
- 若没有共同有效位置，相似度设为 -1.0
- 训练阶段排除时间重叠窗口，排除范围为 [t - L, t + L]
- 输入窗口长度 L = 96

## 历史记忆库大小

**历史记忆库大小不是固定常数，而是训练集窗口数。**

```text
|B_memory| = N_train
```

实现方式：

- 每个训练窗口写入一组 key-value
- key shape = [N_train, L, C]
- value shape = [N_train, H, C_y]
- L = 96
- H = pred_len

论文中可写：

```text
The memory size is set to the number of training windows, i.e.,
|B_memory| = N_train, with one key-value pair stored for each training sample.
```

## 3. 检索与融合参数

具体数值：

- batch size B = 32
- retrieved cases m = 2
- retrieval softmax temperature tau = 1.0
- heuristic retrieval threshold tau_r = 0.55
- heuristic fusion upper bound alpha_max = 0.02
- gate fusion upper bound lambda_max = 0.1
- gate regularization weight = 1e-4
- **lambda_max = 0.1**

最终融合形式：

```text
Y_hat(t) = (1 - lambda_t) * Y_hat_moe(t) + lambda_t * Y_hat_his(t)
0 <= lambda_t <= 0.1
```

因此：

```text
lambda_max = 0.1
```

## 正则项权重

| 参数 | 含义 | 具体数值 |
|---|---|---:|
| lambda_max | 历史检索分支融合权重上界 | 0.1 |
| lambda_s | Student-t component diversity 权重 | 0.001 |
| lambda_b | sample-level router balance 权重 | 0.1 |
| lambda_trend | trend-consistency loss 权重 |  |
| 温度 T |  | 1.0 |
