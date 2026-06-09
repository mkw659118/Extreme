# DARNet 消融实验实现细节

本文档整理了当前 DARNet 代码中已经实现的三组消融实验，分别为取消检索分支、取消 Student-t 状态先验并改用可学习 embedding router、以及严格版去除 MoE 多专家结构。三组实验都以完整 DARNet 作为对照，尽量保持除目标模块之外的训练流程、数据处理方式、损失函数、预测头、评估方式和日志记录一致，从而使实验结果能够更清楚地反映被消融模块本身的贡献。

## 1. w/o Retrieval：取消检索分支

该消融实验用于验证检索记忆库及其预测融合机制对模型性能的贡献。完整 DARNet 在 backbone 得到基础预测 `point_pred` 后，会根据训练集历史窗口构建 key-value 检索库，并在测试或 gate 训练阶段通过输入窗口与历史 key 的相似度检索对应 value，得到 `retrieval_pred`。随后模型使用检索 gate 或启发式融合权重，将 backbone 预测与检索预测进行加权融合。该分支的作用是利用历史相似模式对当前预测进行修正，尤其适合存在重复模式或异常模式可从历史中借鉴的数据场景。

在该消融中，我们新增配置开关 `use_retrieval`，默认值为 `True`，以保持原始模型行为不变。当设置 `--use_retrieval False` 时，模型完全跳过检索相关流程。具体实现包括：在 `modules/DARNet1.py` 中读取 `self.use_retrieval`，关闭时 `construct_index()` 不再申请 key/value memory，`add_key_value()` 不再写入检索库，`retrieval()` 被显式禁止调用，`mark_gate_ready()` 也不会将检索 gate 标记为可用。在 forward 过程中，如果 `use_retrieval=False`，模型直接使用 backbone 输出的 `point_pred`，不会计算 `retrieval_pred`，不会进行 gate fusion 或 heuristic fusion，也不会向辅助损失中加入 retrieval gate 正则项。

训练流程也同步进行了控制。在 `exp/exp_base_DARNet.py` 中，`prepare_retrieval_index()` 在关闭检索时直接返回，不构建训练集检索库；`setup_gate_optimizer()` 在关闭检索时跳过检索 gate 优化器设置；主训练结束后也不会进入 gate training 阶段。因此该实验不仅在推理阶段去掉检索融合，也在训练阶段去掉检索库构建和检索 gate 训练，避免出现“训练了检索分支但测试不用”或“测试不用但仍有检索相关参数参与”的不干净对照。

该实验的运行方式是在原始 DARNet 训练命令中加入：

```bash
--use_retrieval False
```

当前对应脚本为 `script/DARNet_no_retrieval.sh`。该消融可表述为 `w/o Retrieval`，用于回答：历史检索记忆库和检索预测融合是否能够提升 DARNet 的预测性能。

## 2. w/o Student-t Prior：取消 Student-t 状态先验，改用 learned embedding router

该消融实验用于验证 Student-t mixture 状态先验及其 posterior 概率对 MoE 路由的贡献。完整 DARNet 首先使用 `StudentTMixturePrior` 对输入窗口进行状态建模，得到每个样本属于不同隐含状态的概率分布 `q`。随后 `RouterFromEmbeddingPreTrain` 以该 `q` 作为输入，输出 MoE expert 的路由 logits，再经 softmax 和 top-k 得到稀疏专家权重。也就是说，完整模型中的 MoE 路由不是直接从 embedding 学习，而是被 Student-t 状态 posterior 显式引导。除此之外，训练阶段还包含 Student-t prior 的预训练，以及主训练中的状态均衡、dominance penalty、component diversity 和 assignment entropy 等与 Student-t prior 相关的约束。

仅设置 `pretrain_epochs=0` 并不能构成真正的 `w/o Student-t Prior`，因为此时 Student-t prior 在主训练 forward 中仍会计算 `q` 并影响 MoE 路由。因此本消融采用更严格的替代结构：完全关闭 Student-t prior 的路由路径，但保留 MoE 的样本自适应能力。为此新增配置开关 `use_state_prior`，默认值为 `True`。当设置 `--use_state_prior False` 时，模型不再调用 `self.state_prior(prior_x)`，不再使用 Student-t posterior `q` 作为 router 输入，而是使用新增的 `RouterFromEmbeddingFeatures` 直接从输入 embedding 的统计摘要中学习路由。

具体来说，新增 router 从 encoder embedding `x_emb` 中提取三类样本级特征：最后一个时间步表示 `x_emb[:, -1, :]`、时间维均值 `x_emb.mean(dim=1)`、以及时间维标准差 `x_emb.std(dim=1)`。三者拼接后形成 `3 * d_model` 维特征，再通过两层 MLP 输出 `num_experts` 维 router logits。之后的 softmax、top-k、expert 加权融合和 forecast head 与原始 MoE backbone 保持一致。因此该消融并不是把 MoE 路由退化成固定权重，而是保留 learned adaptive routing，只移除 Student-t 状态分布对路由的显式建模和引导。

训练流程中，当 `use_state_prior=False` 时，`RunOnce()` 会跳过 State Prior Pretraining，不再执行 Student-t prior 的预训练循环；`forward()` 中也跳过所有 Student-t 相关的主训练正则项，包括 state balance loss、state dominance loss、component diversity loss 和 assignment entropy loss。为了保持日志和可视化流程兼容，模型仍会记录 `state_probs` 字段，但此时该字段对应 learned router 的 softmax 输出，仅作为路由分布监控使用，不再代表 Student-t posterior。

该实验的运行方式是在原始 DARNet 训练命令中加入：

```bash
--use_state_prior False
```

当前对应脚本为 `script/DARNet_no_state_prior.sh`。该消融可表述为 `w/o Student-t Prior, using learned embedding router`。它回答的问题是：在 MoE 仍然具有样本自适应路由能力的前提下，Student-t 状态先验及其 posterior 引导是否优于普通的 learned embedding router。

## 3. w/o MoE：保留多状态 Student-t prior，去除多专家 backbone

该消融实验用于验证 MoE 多专家结构本身的贡献。需要注意的是，在原始实现中，`num_experts` 同时控制 Student-t mixture 的 component 数量、router 输出维度以及 backbone 中 LSTMExpert 的数量。如果简单设置 `--num_experts 1 --top_k_experts 1`，模型会同时退化为单 Student-t 状态和单 expert，这样消融掉的是“多状态 + 多专家联合结构”，不能严格称为只去掉 MoE。为了实现更干净的 `w/o MoE`，我们将 Student-t 状态数和 MoE expert 数解耦。

具体实现是新增配置 `state_num`。当 `state_num=0` 时，状态数默认跟随 `num_experts`，从而保持原始 full model 的行为完全兼容；当显式指定 `state_num` 时，`StudentTMixturePrior` 的 `num_components` 使用 `state_num`，而 `BackboneMoE` 的 expert 数量继续使用 `num_experts`。同时，`RouterFromEmbeddingPreTrain` 的输入维度改为 `state_num`，输出维度改为 `num_experts`。这样 Student-t prior 可以继续输出多状态 posterior `q`，而 router 将其映射到任意数量的 experts。该设计允许我们设置 `state_num=4, num_experts=1`，从而保留 4 个 Student-t 状态，但 backbone 中只有 1 个 LSTM expert。

严格版 `w/o MoE` 的运行配置为：

```bash
--state_num 4
--num_experts 1
--top_k_experts 1
```

在该配置下，forward 流程为：输入窗口先经过 Student-t prior 得到 4 维状态概率 `state_probs`，router 将 4 维状态概率映射为 1 维 expert logits，softmax 后得到恒为 1 的 expert 权重，backbone 中只包含一个 `LSTMExpert`，因此 `mix_weights` 恒为 `[1.0]`。预测头、检索分支、检索 gate、损失函数和评估流程均保持不变。为了避免单 expert 情况下负载均衡损失产生不合理惩罚，`compute_sample_level_balance_loss()` 在 expert 数量为 1 时返回 0，同时仍记录 expert load 为 `[1.0]`。

该实验对应脚本为 `script/DARNet_wo_moe.sh`。当前验证结果显示，在该配置下模型内部维度为：`state_probs` 形状为 `[B, 4]`，`router_prob` 形状为 `[B, 1]`，`len(backbone.experts)=1`，`mix_weights=[[1.0], ...]`。因此该实验可以严格表述为 `w/o MoE`：保留多状态 Student-t prior，但将多专家 MoE backbone 替换为单专家 backbone。它回答的问题是：在状态建模、检索记忆和其它训练流程保持一致的情况下，多专家结构是否带来额外收益。

## 总结

三组消融实验分别对应 DARNet 中三个关键设计点。`w/o Retrieval` 移除历史相似模式检索与融合，评估检索记忆库的贡献；`w/o Student-t Prior` 移除 Student-t posterior 对 MoE 的显式引导，但保留 learned adaptive router，评估概率状态先验的贡献；`w/o MoE` 解耦状态数与专家数，保留多状态 Student-t prior，仅将 backbone 退化为单 expert，评估多专家结构的贡献。三者互相正交，能够从检索增强、状态先验建模和多专家预测三个层面对完整 DARNet 的有效性进行解释。
