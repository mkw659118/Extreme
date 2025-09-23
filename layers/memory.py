#Author  :   mkw 
#Time    :   2025/09/17 17:27:29
#Desc    :   None

import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleMemory(nn.Module):
    def __init__(self, d_model, mem_size, topk, temperature=0.5, ema_momentum=0.9):
        super().__init__()
        self.d_model = d_model
        self.mem_size = mem_size
        self.topk = topk
        self.temperature = temperature
        self.ema_momentum = ema_momentum

        # 初始化 memory_keys 和 memory_values
        self.memory_keys = torch.randn(mem_size, d_model)  # [mem_size, d_model]
        self.memory_values = torch.randn(mem_size, d_model)  # [mem_size, d_model]

        # 初始化 memory_weights
        self.memory_weights = torch.zeros(mem_size)  # [mem_size]
        
        # 记忆库的指针
        self.memory_ptr = 0  # 当前写入位置

    def read(self, sample_ids, query):
        """
        从记忆库中读取最相似的样本，并确保所有张量在同一设备上。
        """
        # 获取 query 和 memory_keys 的设备（假设它们应该都在同一设备上）
        device = query.device if query.device == self.memory_keys.device else self.memory_keys.device
        
        # 将 query 移动到与 memory_keys 相同的设备
        query = query.to(device)
        
        # 计算查询向量与记忆库中的键之间的余弦相似度
        similarity = F.cosine_similarity(query.unsqueeze(1), self.memory_keys.unsqueeze(0), dim=-1)  # [B, mem_size]

        # 获取最相似的 topk 个记忆
        topk_sim, topk_idx = torch.topk(similarity, self.topk, dim=-1, largest=True)  # [B, topk]

        # 根据 topk_idx 获取相应的 memory_values
        topk_values = self.memory_values[topk_idx]  # [B, topk, d_model]

        return topk_values, topk_sim, topk_idx

    def write(self, sample_ids, k_batch, v_batch):
        """
        将新的样本写入记忆库。
        
        Args:
        - sample_ids: [B] LongTensor，样本唯一ID。
        - k_batch: [B, d_model] 样本的键（查询向量）。
        - v_batch: [B, d_model] 样本的值（记忆内容）。
        """
        batch_size = sample_ids.shape[0]
        
        for i in range(batch_size):
            idx = self.memory_ptr % self.mem_size  # 保证内存大小不超过 mem_size
            
            # 将新的键和值存入记忆库
            self.memory_keys[idx] = k_batch[i]  # [d_model]
            self.memory_values[idx] = v_batch[i]  # [d_model]
            
            # 更新内存写入指针
            self.memory_ptr += 1

            # 使用指数加权平均更新每个样本的权重
            self.memory_weights[idx] = self.ema_momentum * self.memory_weights[idx] + (1 - self.ema_momentum) * 1.0

    def clear(self):
        """
        清空记忆库，重置所有记忆内容。
        """
        self.memory_keys = torch.randn(self.mem_size, self.d_model)  # 重置 memory keys
        self.memory_values = torch.randn(self.mem_size, self.d_model)  # 重置 memory values
        self.memory_weights = torch.zeros(self.mem_size)  # 重置 memory weights
        self.memory_ptr = 0  # 重置内存写入指针
