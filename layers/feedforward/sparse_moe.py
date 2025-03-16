# coding : utf-8
# Author : yuxiang Zeng
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


class SparseDispatcher(object):
    """
    辅助类，用于实现稀疏门控的专家混合（Mixture-of-Experts）。
    主要功能：
      - dispatch：根据门控矩阵将输入分发给各个专家；
      - combine：将各个专家的输出按照对应的门控权重合并回原始顺序。
    """

    def __init__(self, num_experts, gates):
        self._gates = gates  # 门控矩阵，形状 [batch_size, num_experts]
        self._num_experts = num_experts  # 专家数量
        # 找到所有非零门控值的下标，并进行排序
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # 分离出专家索引（第二列）——不再需要 batch 索引
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # 根据排序后的专家索引获取对应的 batch 下标
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # 计算每个专家获得的样本数量
        self._part_sizes = (gates > 0).sum(0).tolist()
        # 根据 batch_index 扩展门控矩阵，使其与分配的样本一一对应
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
        根据非零门控，将输入张量 inp 分发到各个专家。
        Args:
          inp: 输入张量，形状 [batch_size, ...]
        Returns:
          一个列表，每个元素对应一个专家的输入，形状为 [num_samples_for_expert, ...]
        """
        # 先根据 batch_index 选择出分配给专家的样本
        inp_exp = inp[self._batch_index].squeeze(1)
        # 根据各专家获得的样本数量拆分输入
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
        将各个专家的输出按照对应的门控权重合并成最终输出。
        Args:
          expert_out: 一个列表，每个元素是对应专家的输出张量，形状为 [num_samples_for_expert, output_dim]
          multiply_by_gates: 是否使用门控权重对专家输出加权（通常为 True）
        Returns:
          合并后的输出张量，形状为 [batch_size, output_dim]
        """
        # 将各专家的输出拼接成一个大张量
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            # 按对应非零门控值加权
            stitched = stitched.mul(self._nonzero_gates)
        # 创建一个全 0 张量，用于存放合并结果，其大小与原始 batch 保持一致
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1),
                            requires_grad=True, device=stitched.device)
        # 根据 batch_index 将加权后的输出累加到对应的 batch 位置
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """
        将非零的门控值根据各专家拆分开来。
        Returns:
          一个列表，每个元素是对应专家的非零门控值，形状为 [num_samples_for_expert]
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    """
    单层全连接网络（MLP），作为 MoE 中的专家结构。
    包含：
      - 一个线性层从输入映射到隐藏层；
      - 一个 ReLU 激活函数；
      - 一个线性层从隐藏层映射到输出；
      - 最后一个 Softmax 层将输出归一化。
    """

    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入 -> 隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层 -> 输出
        self.relu = nn.ReLU()  # 激活函数
        self.soft = nn.Softmax(dim=1)  # 按行做 Softmax

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class MoE(nn.Module):
    """
    单层 Mixture-of-Experts 模块。
    每个专家为一个简单的 MLP，通过稀疏门控机制选择部分专家参与计算。

    参数：
      - input_size: 输入维度
      - output_size: 输出维度
      - num_experts: 专家数量
      - hidden_size: 专家内部 MLP 的隐藏层维度
      - noisy_gating: 是否在训练时添加噪声（默认 True）
      - k: 每个样本选择的专家数（top-k）
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # 构建专家网络，每个专家都是一个 MLP
        self.experts = nn.ModuleList([
            MLP(self.input_size, self.output_size, self.hidden_size)
            for _ in range(self.num_experts)
        ])
        # 门控权重参数，负责将输入映射到各个专家的打分
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # 用于计算噪声标准差的参数
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()  # 保证噪声标准差为正
        self.softmax = nn.Softmax(dim=1)  # 计算门控概率
        # 注册用于正态分布的均值和标准差缓冲区
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        # 保证选择的专家数不大于专家总数
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """
        计算样本的变异系数平方（coefficient of variation squared）。
        用作负载均衡损失，鼓励各专家的激活分布更均匀。
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """
        根据门控矩阵计算每个专家的负载（即有多少样本被分配到该专家）。
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        辅助函数：计算在加入噪声后，各个专家进入 top-k 的概率，
        便于反向传播负载均衡损失。
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)  # m 通常为 k+1
        top_values_flat = noisy_top_values.flatten()
        # 计算每个样本中第 k 大值的阈值位置
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        # 对于未进入 top-k 的情况，取前一位作为阈值
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        采用 noisy top-k 机制计算门控权重。
        Args:
          x: 输入张量，形状 [batch_size, input_size]
          train: 布尔值，指示是否处于训练模式（训练时添加噪声）
          noise_epsilon: 用于数值稳定性的常数
        Returns:
          gates: 经过稀疏化处理的门控矩阵，形状 [batch_size, num_experts]
          load: 每个专家的负载张量，形状 [num_experts]
        """
        # 计算“干净”的 logits：输入乘以门控权重矩阵
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            # 计算噪声标准差
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon)
            # 在 logits 中加入噪声
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 使用 softmax 将 logits 转换为概率
        logits = self.softmax(logits)
        # 对每个样本选择 top-(k+1) 个专家
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        # 只保留前 k 个专家
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        # 对选中的 k 个专家的概率进行归一化，使其和为 1
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
        # 创建一个与 logits 同形状的全 0 张量
        zeros = torch.zeros_like(logits, requires_grad=True)
        # 将归一化后的 top-k 权重散布到对应专家的位置上，形成稀疏门控矩阵
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 根据是否处于训练模式以及专家数量判断负载计算方式
        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """
        MoE 层的前向传播。
        Args:
          x: 输入张量，形状 [batch_size, input_size]
          loss_coef: 负载均衡损失的缩放系数
        Returns:
          y: 输出张量，形状 [batch_size, output_size]
          loss: 负载均衡损失，鼓励各专家均匀使用
        """
        # 计算门控权重和专家负载
        gates, load = self.noisy_top_k_gating(x, self.training)
        # 计算每个专家的总激活强度（importance）
        importance = gates.sum(0)
        # 计算负载均衡损失：利用重要性和负载的变异系数平方
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        # 根据门控矩阵将输入分发给各个专家
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        # expert_gates 可用于进一步调试或分析，此处未直接使用
        expert_gates = dispatcher.expert_to_gates()
        # 分别通过每个专家计算输出
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        # 将各个专家的输出合并为最终输出
        y = dispatcher.combine(expert_outputs)
        return y, loss


class MultiLayerMoE(nn.Module):
    """
    多层 MoE 模型，通过单个 for 循环构建所有层：
      - 如果只有一层，则构造 MoE(input_size -> output_size)；
      - 如果有多层，则：
           * 第一层：MoE(input_size -> hidden_size)
           * 中间层：MoE(hidden_size -> hidden_size)
           * 最后一层：MoE(hidden_size -> output_size)
    各层的负载均衡损失会在前向传播时累加。
    """

    def __init__(self, num_layers, input_size, hidden_size, output_size, num_experts, expert_hidden_size, noisy_gating=True, k=4):
        super(MultiLayerMoE, self).__init__()
        layers = []
        # 通过 for 循环构建所有层
        for i in range(num_layers):
            # 第一层使用 input_size 作为输入维度，其余层使用 hidden_size
            in_dim = input_size if i == 0 else hidden_size
            # 最后一层输出维度为 output_size，其它层输出为 hidden_size
            out_dim = output_size if i == num_layers - 1 else hidden_size
            layers.append(MoE(in_dim, out_dim, num_experts, expert_hidden_size, noisy_gating, k))
        # 将各层封装到 ModuleList 中，便于管理
        self.layers = nn.ModuleList(layers)

    def forward(self, x, loss_coef=1e-2):
        """
        多层 MoE 模型的前向传播，每层的负载均衡损失累加。
        Args:
          x: 输入张量，形状 [batch_size, input_size]
          loss_coef: 每层负载均衡损失的缩放系数
        Returns:
          x: 最终输出张量
          total_loss: 累积的负载均衡损失
        """
        total_loss = 0.
        # 顺序遍历各层，逐层前向传播
        for layer in self.layers:
            x, loss = layer(x, loss_coef)
            total_loss += loss
        return x, total_loss


# 测试多层 MoE 模型
if __name__ == '__main__':
    # 参数设置
    input_size = 16  # 输入数据的特征维度
    hidden_size = 32  # 隐藏层（中间层）的维度
    output_size = 10  # 输出数据的特征维度
    num_experts = 8  # 每个 MoE 层中专家的数量
    expert_hidden_size = 32  # 专家内部 MLP 的隐藏层维度
    batch_size = 5  # 每个 batch 中的样本数量
    num_layers = 3  # MoE 层的层数

    # 实例化多层 MoE 模型
    model = MultiLayerMoE(num_layers, input_size, hidden_size, output_size,
                          num_experts, expert_hidden_size, noisy_gating=True, k=4)
    model.train()  # 设置为训练模式

    # 生成随机输入样例
    sample_input = torch.randn(batch_size, input_size)

    # 前向传播
    output, total_loss = model(sample_input)

    print("输出 shape：", output.shape)
    print("累积负载平衡损失：", total_loss)