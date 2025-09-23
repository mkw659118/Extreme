#Author  :   mkw 
#Time    :   2025/09/17 17:27:29
#Desc    :   None

import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleMemory(nn.Module):
    # 初始化部分增加设备属性
    def __init__(self, d_model, mem_size, topk, temperature=0.5, ema_momentum=0.9):
        super().__init__()
        self.d_model = d_model
        self.mem_size = mem_size
        self.topk = topk
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化 memory_keys 和 memory_values，并指定设备
        self.memory_keys = torch.randn(mem_size, d_model, device=self.device)
        self.memory_values = torch.randn(mem_size, d_model, device=self.device)
        self.memory_weights = torch.zeros(mem_size, device=self.device)
        self.memory_ptr = 0

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
        batch_size = sample_ids.shape[0]
        
        for i in range(batch_size):
            idx = self.memory_ptr % self.mem_size
            
            # 确保张量在同一设备上
            k = k_batch[i].to(self.device)
            v = v_batch[i].to(self.device)
            
            # 统一调整为一维张量
            k = k.flatten()
            v = v.flatten()
            
            # 确保维度匹配
            if k.shape != self.memory_keys[idx].shape:
                k = k.resize_(self.memory_keys[idx].shape)
            if v.shape != self.memory_values[idx].shape:
                v = v.resize_(self.memory_values[idx].shape)
            
            self.memory_keys[idx] = k
            self.memory_values[idx] = v
            self.memory_ptr += 1
            self.memory_weights[idx] = self.ema_momentum * self.memory_weights[idx] + (1 - self.ema_momentum) * 1.0

    def clear(self):
        """
        清空记忆库，重置所有记忆内容。
        """
        self.memory_keys = torch.randn(self.mem_size, self.d_model)  # 重置 memory keys
        self.memory_values = torch.randn(self.mem_size, self.d_model)  # 重置 memory values
        self.memory_weights = torch.zeros(self.mem_size)  # 重置 memory weights
        self.memory_ptr = 0  # 重置内存写入指针
