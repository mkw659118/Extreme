import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, output_dim)
        )

    def forward(self, x):
        return self.layer(x)  # x 形状: (batch, seq_len, d_model)


class GatingNetwork(nn.Module):
    def __init__(self, d_model, num_experts, k=2, noise_std=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # Top-k 选择
        self.noise_std = noise_std  # 训练时添加噪声
        self.gate = nn.Linear(d_model, num_experts)  # (batch, seq_len, num_experts)

    def forward(self, x):
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std  # 训练时加入噪声
            logits += noise
        weights = F.softmax(logits, dim=-1)  # 转化为概率 (batch, seq_len, num_experts)
        topk_weights, topk_indices = torch.topk(weights, self.k, dim=-1)  # 选择 top-k (batch, seq_len, k)
        return topk_weights, topk_indices, weights


class MoE(nn.Module):
    def __init__(self, d_model, d_ff, d_out, num_shared_experts=2, num_routed_experts=4, topk=2, noise_std=0.1):
        super().__init__()
        self.topk = topk
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.shared_expert_modules = nn.ModuleList([Expert(d_model, d_ff, d_out) for _ in range(num_shared_experts)])
        self.routed_expert_modules = nn.ModuleList([Expert(d_model, d_ff, d_out) for _ in range(num_routed_experts)])
        self.gating_network = GatingNetwork(d_model, num_routed_experts, topk, noise_std)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        # (batch, seq_len, k), (batch, seq_len, k), (batch, seq_len, num_experts)
        topk_weights, topk_indices, all_weights = self.gating_network(x)
        self.all_weights = all_weights  # 记录专家权重

        # 计算共享专家输出
        shared_outputs = sum(expert(x) for expert in self.shared_expert_modules) / self.num_shared_experts  # (batch, seq_len, d_out)

        # 计算 routed expert 输出（优化）
        # (batch, seq_len, k) -> (batch, seq_len, k, d_model)
        expanded_x = x.unsqueeze(2).expand(-1, -1, self.topk, -1)  # 复制 x 以匹配 k 个专家

        # 获取专家索引
        expert_indices = topk_indices.view(batch_size, seq_len, self.topk)  # (batch, seq_len, k)

        # 转换专家索引为 one-hot 向量
        expert_one_hot = F.one_hot(expert_indices, num_classes=self.num_routed_experts).float()  # (batch, seq_len, k, num_experts)

        # 批量处理专家计算
        expert_outputs = torch.stack([expert(expanded_x) for expert in self.routed_expert_modules], dim=-2)  # (batch, seq_len, num_experts, d_out)

        # 根据 top-k 选择相应专家的输出
        routed_outputs = (expert_outputs * expert_one_hot.unsqueeze(-1)).sum(dim=-2)  # (batch, seq_len, k, d_out)

        # 按权重加权求和
        routed_outputs = (routed_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)  # (batch, seq_len, d_out)

        # 最终输出
        output = shared_outputs + routed_outputs  # (batch, seq_len, d_out)
        return output

    # def load_balancing_loss(self):
    #     """
    #     计算负熵负载均衡损失，鼓励专家均匀使用
    #     """
    #     expert_probs = self.all_weights.mean(dim=(0, 1))  # 在 batch 和 seq_len 维度上取平均
    #     load_balancing_loss = (expert_probs * torch.log(expert_probs + 1e-10)).sum()
    #     return -load_balancing_loss  # 负熵，用于均衡专家使用

    def load_balancing_loss(self):
        expert_probs = self.all_weights.mean(dim=(0, 1))  # shape: (num_experts,)
        num_experts = expert_probs.shape[0]
        # KL 散度：sum( p_i * log(p_i / (1/N)) ) = sum( p_i * (log(p_i) + log(N)) )
        # 当 p_i = 1/N 时，loss 为0
        loss = (expert_probs * (torch.log(expert_probs + 1e-10) + torch.log(torch.tensor(num_experts, dtype=torch.float)))).sum()
        return loss

# 示例用法
if __name__ == "__main__":
    model = MoE(d_model=10, d_ff=512, d_out=5, num_shared_experts=2, num_routed_experts=4, topk=2, noise_std=0.1)
    x = torch.rand(4, 20, 10)  # batch_size=4, seq_len=20, d_model=10
    output = model(x)
    print(output.shape)  # 输出应为 (4, 20, 5)
    print("Load balancing loss:", model.load_balancing_loss().item())