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
