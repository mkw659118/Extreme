# coding : utf-8
# Author : yuxiang Zeng

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineGrainedFFN(nn.Module):
    """
    单个小专家 FFN 模块，输入维度为 input_dim，
    中间维度为 ffn_dim_small，输出维度恢复为 input_dim。
    """

    def __init__(self, input_dim, ffn_dim_small):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, ffn_dim_small)
        self.fc2 = nn.Linear(ffn_dim_small, input_dim)
        self.activation = nn.ReLU()  # 可选：nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class FineGrainedMoE(nn.Module):
    """
    细粒度 MoE 层：
      - 将标准 MoE 中的 N 个专家，每个 FFN 的中间维度为 ffn_dim，
        分割为 m 个小专家（每个小专家中间维度为 ffn_dim / m），
        整体专家数变为 m * N，但总参数量与标准 FFN 保持一致。
      - 同时将激活的专家数从 K 扩展为 m*K 以保持计算成本一致。

    输入：
      x: Tensor，形状为 (batch_size, input_dim)

    输出：
      output: Tensor，形状为 (batch_size, input_dim)
             输出为残差连接：x + 加权求和所有激活专家的输出。
    """

    def __init__(self, input_dim, ffn_dim, N, m, K):
        """
        参数说明：
          input_dim: 输入的隐藏维度。
          ffn_dim: 标准 FFN 中间层维度（例如：4 * input_dim）。
          N: 标准 MoE 中专家数量。
          m: 分割因子（每个专家分割为 m 个小专家）。
          K: 标准路由时激活的专家数；细粒度策略下，激活专家数为 m*K。
        """
        super().__init__()
        self.input_dim = input_dim
        self.N = N
        self.m = m
        self.K = K  # 原始 MoE 中激活专家个数
        self.total_experts = m * N  # 分割后总专家数量
        # 每个小专家的中间层维度
        assert ffn_dim % m == 0, "ffn_dim 必须能被 m 整除"
        self.ffn_dim_small = ffn_dim // m

        # 构造 m*N 个小专家
        self.experts = nn.ModuleList([
            FineGrainedFFN(input_dim, self.ffn_dim_small)
            for _ in range(self.total_experts)
        ])

        # 门控网络：对每个输入 token，计算与每个专家对应的得分
        # 这里使用一个简单的线性变换，相当于与每个专家对应的门控向量做内积
        self.gate = nn.Linear(input_dim, self.total_experts, bias=False)

    def forward(self, x):
        """
        x: Tensor, 形状为 (batch_size, input_dim)
        """
        # 计算门控得分 logits: (batch_size, total_experts)
        logits = self.gate(x)
        # 对每个样本计算 softmax 得到概率分布 s_{i,t}
        gate_probs = F.softmax(logits, dim=-1)

        # 对每个样本选取 top (m*K) 个专家
        topk = self.m * self.K
        # 得到 topk 的分数和对应索引
        topk_values, topk_indices = torch.topk(gate_probs, topk, dim=-1)

        # 构造掩码，将非 topk 的位置置 0
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(1, topk_indices, 1.0)
        # 仅保留 topk 专家的概率，其它置 0
        gate_probs = gate_probs * mask

        # 计算每个专家的输出
        # 这里对所有专家并行计算，得到形状 (batch_size, total_experts, input_dim)
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (batch_size, input_dim)
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 将门控概率乘到对应的专家输出上
        # gate_probs: (batch_size, total_experts) -> (batch_size, total_experts, 1)
        weighted_expert_outputs = gate_probs.unsqueeze(-1) * expert_outputs

        # 将所有专家输出加和，并加上残差连接
        moe_output = weighted_expert_outputs.sum(dim=1)  # (batch_size, input_dim)
        output = x + moe_output

        return output


# ===== 示例用法 =====
if __name__ == "__main__":
    # 假设 hidden_size 为 512，标准 FFN 中间维度为 2048
    input_dim = 512
    ffn_dim = 2048
    N = 16  # 标准专家数
    m = 4  # 分割因子，每个专家分成4个小专家，故总专家数为 64
    K = 2  # 标准路由时激活 2 个专家，细粒度下激活 2*4=8 个专家

    # 构造细粒度 MoE 层
    moe_layer = FineGrainedMoE(input_dim, ffn_dim, N, m, K)

    # 假设 batch_size 为 8
    x = torch.randn(8, input_dim)
    output = moe_layer(x)
    print("Output shape:", output.shape)  # 应输出 (8, 512)