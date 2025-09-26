import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SampleMemory(nn.Module):
    # 初始化部分增加设备属性，并添加ID生成器
    def __init__(self, d_model, mem_size, topk, temperature=0.5, ema_momentum=0.9):
        super().__init__()
        self.d_model = d_model
        self.mem_size = mem_size
        self.topk = topk
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化记忆库参数，并指定设备
        self.memory_keys = torch.randn(mem_size, d_model, device=self.device)
        self.memory_values = torch.randn(mem_size, d_model, device=self.device)
        self.memory_weights = torch.zeros(mem_size, device=self.device)
        self.memory_sample_ids = torch.full((mem_size,), -1, dtype=torch.long, device=self.device)  # 存储样本ID
        self.memory_ptr = 0

        # 添加样本ID生成器
        self.sample_id_counter = 0

    def generate_sample_id(self):
        """生成一个唯一的样本ID"""
        self.sample_id_counter += 1
        return self.sample_id_counter

    def read(self, sample_ids, query):
        """
        从记忆库中读取与样本相关的最相似记忆
        sample_ids: 样本ID列表，用于筛选相关记忆 [B]
        query: 查询向量 [B, d_model]
        """
        device = self.device
        query = query.to(device)
        sample_ids = sample_ids.to(device)
        
        batch_size = query.shape[0]
        all_topk_values = []
        all_topk_sim = []
        all_topk_idx = []
        
        for i in range(batch_size):
            sid = sample_ids[i]
            
            # 找到与当前样本ID相关的记忆索引
            related_mask = (self.memory_sample_ids == sid)
            related_indices = torch.nonzero(related_mask).squeeze()
            
            # 如果没有相关记忆或相关记忆太少，使用所有记忆
            if related_indices.numel() <= self.topk or related_indices.numel() == 0:
                candidate_keys = self.memory_keys
                candidate_indices = torch.arange(self.mem_size, device=device)
            else:
                candidate_keys = self.memory_keys[related_indices]
                candidate_indices = related_indices
            
            # 计算查询与候选记忆的余弦相似度
            similarity = F.cosine_similarity(
                query[i].unsqueeze(0), 
                candidate_keys, 
                dim=-1
            )
            
            # 获取最相似的topk个记忆
            topk_sim, topk_sub_idx = torch.topk(
                similarity, 
                min(self.topk, len(similarity)), 
                dim=-1, 
                largest=True
            )
            
            # 转换为全局索引
            topk_idx = candidate_indices[topk_sub_idx]
            topk_values = self.memory_values[topk_idx]
            
            # 填充不足的部分（如果候选记忆少于topk）
            if len(topk_values) < self.topk:
                pad_size = self.topk - len(topk_values)
                topk_values = torch.cat([topk_values, torch.zeros(pad_size, self.d_model, device=device)])
                topk_sim = torch.cat([topk_sim, torch.full((pad_size,), -1.0, device=device)])
                topk_idx = torch.cat([topk_idx, torch.full((pad_size,), -1, dtype=torch.long, device=device)])
            
            all_topk_values.append(topk_values.unsqueeze(0))
            all_topk_sim.append(topk_sim.unsqueeze(0))
            all_topk_idx.append(topk_idx.unsqueeze(0))
        
        return (
            torch.cat(all_topk_values, dim=0),  
            torch.cat(all_topk_sim, dim=0),     
            torch.cat(all_topk_idx, dim=0)      
        )

    def write(self, sample_ids, k_batch, v_batch):
        """
        写入记忆库，同时记录样本ID
        sample_ids: 样本ID列表 [B]
        k_batch: 键向量批次 [B, ..., d_model]
        v_batch: 值向量批次 [B, ..., d_model]
        """
        batch_size = sample_ids.shape[0]
        sample_ids = sample_ids.to(self.device)
        
        for i in range(batch_size):
            sid = self.generate_sample_id()
            
            idx = self.memory_ptr % self.mem_size
            
            k = k_batch[i].to(self.device).flatten()
            v = v_batch[i].to(self.device).flatten()
            
            if k.shape[0] != self.d_model:
                k = F.adaptive_avg_pool1d(k.unsqueeze(0), self.d_model).squeeze()
            if v.shape[0] != self.d_model:
                v = F.adaptive_avg_pool1d(v.unsqueeze(0), self.d_model).squeeze()
            
            # 写入记忆库
            self.memory_keys[idx] = k
            self.memory_values[idx] = v
            self.memory_sample_ids[idx] = sid  # 记录样本ID
            self.memory_weights[idx] = self.ema_momentum * self.memory_weights[idx] + (1 - self.ema_momentum) * 1.0
            
            self.memory_ptr += 1

    def clear(self):
        """清空记忆库，重置所有记忆内容"""
        self.memory_keys = torch.randn(self.mem_size, self.d_model, device=self.device)
        self.memory_values = torch.randn(self.mem_size, self.d_model, device=self.device)
        self.memory_weights = torch.zeros(self.mem_size, device=self.device)
        self.memory_sample_ids = torch.full((self.mem_size,), -1, dtype=torch.long, device=self.device)
        self.memory_ptr = 0

    def get_sample_memory(self, sample_id):
        """获取特定样本的所有记忆"""
        sample_id = sample_id.to(self.device)
        mask = (self.memory_sample_ids == sample_id)
        return (
            self.memory_keys[mask],
            self.memory_values[mask],
            self.memory_weights[mask]
        )
