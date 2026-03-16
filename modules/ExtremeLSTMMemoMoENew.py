# #Author  :   mkw
# #Time    :   2025/09/17 10:50:52
# #Desc    :   ExtremeLSTMMemo (MoE on LSTM Layer + Standard MoE Head)

# import time

# from layers.embedding import DataEmbedding
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.att.cross_attention import CrossAttention

# class MoEExpert(nn.Module):
#     def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.3):
#         super().__init__()
#         hidden = hidden or (2 * d_model)
#         self.fc1 = nn.Linear(d_model, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, d_model)
#         self.act = nn.GELU()
#         self.drop1 = nn.Dropout(dropout)
#         self.drop2 = nn.Dropout(dropout)
#         self.drop3 = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.drop1(self.act(self.fc1(x)))
#         x = self.drop2(self.act(self.fc2(x)))
#         x = self.drop3(self.fc3(x))
#         return x

# class StandardMoEHead(nn.Module):
#     def __init__(self, d_model: int, num_experts: int = 3, top_k: int = 2, dropout: float = 0.3):
#         super().__init__()
#         self.d_model = d_model
#         self.num_experts = num_experts
#         self.top_k = top_k
        
#         # 1. 专家池：每个专家输出维度=d_model
#         self.experts = nn.ModuleList([
#             MoEExpert(d_model, hidden=2*d_model, dropout=dropout) 
#             for _ in range(num_experts)
#         ])
        
#         # 2. 最后统一映射到1维（所有专家共享）
#         self.final_proj = nn.Linear(d_model, 1)

#     def forward(self, x, router_probs, topk_experts):
        
#         B, pred_len, D = x.shape
        
#         # 初始化MoE输出（维度保持d_model）
#         moe_out = torch.zeros_like(x)  # [B, pred_len, d_model]
        
#         # 遍历top-k专家，按索引选择并加权融合
#         for k in range(self.top_k):
#             # 获取第k个专家的索引和权重
#             expert_idx = topk_experts[:, k]  # [B]
#             weight = router_probs[:, k].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            
#             # 按索引选择对应专家计算
#             for e in range(self.num_experts):
#                 # 找到选中当前专家的样本掩码
#                 mask = (expert_idx == e)
#                 if not mask.any():
#                     continue
                
#                 # 专家前向计算（输出d_model） + 加权累加
#                 expert_feat = self.experts[e](x[mask])  # [mask_B, pred_len, d_model]
#                 moe_out[mask] += expert_feat * weight[mask]
        
#         # 最后统一映射到1维
#         final_out = self.final_proj(moe_out)  # [B, pred_len, 1]
#         return final_out
    
# class SampleRouterFromX(nn.Module):
#     def __init__(self, c_in: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
#         super().__init__()
#         self.c_in = c_in
#         in_dim = 3 * c_in  # std + max_abs + last
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, num_experts),
#         )

#     def forward(self, x):
#         std = x.std(dim=1, unbiased=False)  # [B, C]
#         max_abs = x.abs().amax(dim=1)  # [B, C]
#         last = x[:, -1, :]  # [B, C]
#         feat = torch.cat([std, max_abs, last], dim=-1)  # [B, 3C]
#         return self.net(feat)

# # 分层专家LSTM（保持不变）
# class ExpertLSTM(nn.Module):
#     """为每一层LSTM设计独立的专家分支，支持top-k加权融合"""
#     def __init__(self, d_model: int, num_layers: int, num_experts: int, 
#                  top_k_experts: int = 2, dropout: float = 0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.num_layers = num_layers
#         self.num_experts = num_experts
#         self.top_k_experts = top_k_experts
#         self.dropout = dropout

#         # 1. 为每一层LSTM创建多个专家（单层LSTM）
#         self.layer_experts = nn.ModuleList([
#             nn.ModuleList([
#                 nn.LSTM(
#                     input_size=d_model,
#                     hidden_size=d_model,
#                     num_layers=1,
#                     batch_first=True,
#                     dropout=dropout
#                 ) for _ in range(num_experts)
#             ]) for _ in range(num_layers)
#         ])

#         # 2. 每层独立的路由器
#         self.layer_routers = nn.ModuleList([
#             SampleRouterFromX(
#                 c_in=d_model, 
#                 num_experts=num_experts, 
#                 hidden=d_model, 
#                 dropout=dropout
#             ) for _ in range(num_layers)
#         ])

#     def forward(self, x, hc=None):
#         B, L, D = x.shape
#         h_n = torch.zeros(self.num_layers, B, D, device=x.device)
#         c_n = torch.zeros(self.num_layers, B, D, device=x.device)

#         layer_input = x
#         for layer_idx in range(self.num_layers):
#             # a. 当前层的路由器：计算专家权重 [B, num_experts]
#             router_logits = self.layer_routers[layer_idx](layer_input)
#             router_prob = F.softmax(router_logits, dim=-1)

#             # b. 选择top-k专家并归一化权重
#             topk_probs, topk_experts = torch.topk(router_prob, k=self.top_k_experts, dim=-1)
#             mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # [B, k]

#             # c. 初始化当前层输出
#             layer_output = torch.zeros_like(layer_input)
#             layer_h = torch.zeros(1, B, D, device=x.device)
#             layer_c = torch.zeros(1, B, D, device=x.device)

#             # d. 遍历top-k专家，加权融合输出
#             for k in range(self.top_k_experts):
#                 expert_id = topk_experts[:, k]
#                 weight = mix_weights[:, k].unsqueeze(-1).unsqueeze(-1)

#                 for expert_idx in range(self.num_experts):
#                     mask = (expert_id == expert_idx)
#                     if not mask.any():
#                         continue

#                     expert_lstm = self.layer_experts[layer_idx][expert_idx]
#                     expert_hc = None
#                     if hc is not None:
#                         expert_h = hc[0][layer_idx:layer_idx+1, mask, :]
#                         expert_c = hc[1][layer_idx:layer_idx+1, mask, :]
#                         expert_hc = (expert_h, expert_c)

#                     expert_out, (expert_h, expert_c) = expert_lstm(layer_input[mask], expert_hc)
#                     layer_output[mask] += expert_out * weight[mask]
#                     layer_h[:, mask, :] += expert_h * weight[mask].squeeze(1)
#                     layer_c[:, mask, :] += expert_c * weight[mask].squeeze(1)

#             # e. 更新当前层的隐藏状态
#             h_n[layer_idx:layer_idx+1, :, :] = layer_h
#             c_n[layer_idx:layer_idx+1, :, :] = layer_c

#             # f. 作为下一层的输入
#             layer_input = layer_output

#         return layer_output, (h_n, c_n)

# # ========== 主模型：替换为标准MoE Head ==========
# class ExtremeLSTMMemo(nn.Module):
#     def __init__(
#         self,
#         c_in: int ,
#         seq_len: int,
#         pred_len: int,                
#         d_model: int,
#         e_layers: int, 
#         d_layers: int, 
#         config=None,
#     ):
#         super().__init__()
#         self.config = config
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.d_model = d_model
#         self.c_in = c_in
#         self.dropout = self.config.dropout
#         self.device = self.config.device

#         # -------- expert definition --------
#         self.num_experts = 10
#         self.retrieval_num = 4
#         self.top_k_experts = 2  # MoE统一用top-2
#         self.retrieval_stride = 1


        
#         # -------- Embedding + pred tokens --------
#         self.enc_embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
#         self.dec_embedding = DataEmbedding(c_in=5, d_model=d_model, dropout=self.dropout)

#         # -------- 专家LSTM编码器/解码器 --------
#         self.encoder = ExpertLSTM(
#             d_model=d_model,
#             num_layers=e_layers,
#             num_experts=self.num_experts,
#             top_k_experts=self.top_k_experts,
#             dropout=self.dropout
#         )
#         self.decoder = ExpertLSTM(
#             d_model=d_model,
#             num_layers=d_layers,
#             num_experts=self.num_experts,
#             top_k_experts=self.top_k_experts,
#             dropout=self.dropout    
#         )

#         self.post_norm = nn.RMSNorm(d_model)
#         self.xattn = CrossAttention(d_model)

#         # -------- router --------
#         router_hidden = self.d_model
#         router_dropout = self.dropout
#         self.router = SampleRouterFromX(c_in=c_in, num_experts=self.num_experts, hidden=router_hidden, dropout=router_dropout)

#         # -------- 核心修正：标准MoE Head --------
#         self.moe_head = StandardMoEHead(
#             d_model=d_model,
#             num_experts=self.num_experts,
#             top_k=self.top_k_experts,
#             dropout=self.dropout
#         )
        
#         self.fuse_proj = nn.Linear(2 * d_model, d_model)
#         self.out_proj = nn.Linear(d_model, 1)

#          # -------- 活跃度损失相关 --------
#         self.expert_activity_loss_weight = 0.1  # 活跃度惩罚的权重，可以根据实验调整

#     def expert_activity_loss(self, router_prob):
#         """
#         计算专家活跃度损失，惩罚活跃度过低的专家。
#         router_prob: [B, num_experts]，即每个样本选择各个专家的概率
#         """
#         # 计算每个专家的激活频率：对于每个专家，计算其在整个batch中的激活概率
#         expert_activity = torch.mean(router_prob, dim=0)  # [num_experts]
        
#         # 我们希望所有专家的激活频率都接近均匀，因此可以计算活跃度的方差作为惩罚项
#         activity_variance = torch.var(expert_activity, dim=-1)
        
#         # 返回一个惩罚项
#         return activity_variance

#     def construct_index(self, num):
#         key_len = self.seq_len
#         self.keys = torch.zeros(num, key_len, self.c_in, device=self.device)
#         self.values = torch.zeros(num, self.pred_len, 5, device=self.device)
#         self.index = 0

#     @torch.no_grad()
#     def add_key_value(self, x_enc, y, index):
#         bs = x_enc.shape[0]
#         x_key = x_enc
#         self.keys[index, :, :] = x_key
#         self.values[index, :, :] = y
#         self.index += bs
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def retrieval(self, x, index):
#         bs = x.shape[0]
#         k = self.retrieval_num

#         queries = x                         # [B, seq_len, x_dim]
#         keys = self.keys[:self.index]       # [N, seq_len, x_dim]
#         values = self.values[:self.index]   # [N, pred_len, y_dim]

#         # 相似度: [B, N]
#         dis = self.cosine_similarity(queries, keys)

#         # 训练时屏蔽自身附近样本
#         if self.training:
#             self_range = torch.arange(-self.seq_len, self.seq_len + 1, device=x.device).unsqueeze(0)   # [1, 2*seq_len+1]
#             invalid_index = index.unsqueeze(1) + self_range
#             invalid_index = invalid_index // self.retrieval_stride
#             invalid_index[invalid_index < 0] = 0
#             invalid_index[invalid_index >= self.index] = self.index - 1

#             row_idx = torch.arange(bs, device=x.device).unsqueeze(1).repeat(1, invalid_index.size(1))
#             dis[row_idx, invalid_index] = -100.0

#         # top-k
#         dis_topk, indices_topk = torch.topk(dis, dim=1, k=k)   # [B, k]
#         sims = dis_topk                                         # [B, k]
#         probs_topk = torch.softmax(dis_topk, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, k, 1, 1]

#         # 从 values 取出 top-k: [B, k, pred_len, y_dim]
#         retrieved_values = values[indices_topk]   # 高级索引，直接可用

#         # 加权融合: [B, pred_len, y_dim]
#         output = torch.sum(probs_topk * retrieved_values, dim=1)

#         return output, sims, 0

    
#     def cosine_similarity(self, queries, keys):
#         """
#         queries: [B, L, C]
#         keys:    [N, L, C]
#         return:  [B, N]
#         """
#         if len(queries.shape) == 3:
#             B = queries.size(0)
#             N = keys.size(0)

#             queries = queries.reshape(B, -1)   # [B, L*C]
#             keys = keys.reshape(N, -1)         # [N, L*C]

#             q_norm = F.normalize(queries, p=2, dim=-1)
#             k_norm = F.normalize(keys, p=2, dim=-1)
#             return torch.matmul(q_norm, k_norm.t())   # [B, N]

#         elif len(queries.shape) == 2:
#             q_norm = F.normalize(queries, p=2, dim=-1)
#             k_norm = F.normalize(keys, p=2, dim=-1)
#             return torch.matmul(q_norm, k_norm.t())

#         else:
#             raise ValueError(f"Unsupported query shape: {queries.shape}")
        
#     def forward(self, x, x_mark=None, dec_input=None, sample_ids=None, mode='train'):
#         B = x.size(0)
#         # ---------------- routing ----------------
#         router_logits = self.router(x)                       # [B, num_experts]
#         router_prob = torch.softmax(router_logits, dim=-1)    # [B, num_experts]
     
#         # ---------------- embedding ----------------
#         x_emb_hist = self.enc_embedding(x)                        # [B, seq_len, d_model]

#         # =========================================================
#         # LSTM backbone: 专家LSTM编码器 + 专家LSTM解码器
#         # =========================================================
#         enc_out, (h_n, c_n) = self.encoder(x_emb_hist)              # [B, seq_len, d_model]
#         dec_input = self.dec_embedding(dec_input)
#         dec_out, _ = self.decoder(dec_input, (h_n, c_n))             # [B, pred_len, d_model]
#         dec_out = dec_out[:, -self.pred_len:, :]
        
#         # 交叉注意力 + 特征融合（输出仍为d_model）
#         ctx, _ = self.xattn(dec_out, enc_out, enc_out)
#         fused = torch.cat([dec_out, ctx], dim=-1)                  # [B, pred_len, 2*d_model]
#         fused = self.fuse_proj(fused)                              # [B, pred_len, d_model]
#         final_shared = self.post_norm(fused)                        # [B, pred_len, d_model]
#         # y = self.out_proj(final_shared)

#         # =========================================================
#         # 核心修正：标准MoE Head计算
#         # =========================================================
#         # 1. 选择top-k专家并归一化权重
#         topk_result = torch.topk(router_prob, k=self.top_k_experts, dim=-1)
#         topk_probs = topk_result.values     # [B, top_k]
#         topk_experts = topk_result.indices  # [B, top_k]
#         mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # [B, top_k]

#         # 2. 标准MoE前向：专家特征变换（d_model） + 最后映射到1维
#         y = self.moe_head(final_shared, mix_weights, topk_experts)  # [B, pred_len, 1]

#         if mode == 'test':
#             print("Retrieval in test mode...........")
#             retrieval_results, sims, t = self.retrieval(x, sample_ids) # 检索得到的结果
#             print("Retrieval end...........")
           
#             # sim_mean = torch.mean(sims, dim=-1).unsqueeze(-1)  # [bs, 1, 1]
#             # dynamic_alpha = 0.3 * (sim_mean - sim_mean.min()) / (sim_mean.max() - sim_mean.min() + 1e-8)
#             # y = (1 - dynamic_alpha) * y + dynamic_alpha * retrieval_results
#             sim_mean = torch.mean(sims, dim=-1, keepdim=True)   # [B, 1]
#             dynamic_alpha = 0.05 * (sim_mean - sim_mean.min()) / (sim_mean.max() - sim_mean.min() + 1e-8)
#             dynamic_alpha = dynamic_alpha.unsqueeze(-1)         # [B, 1, 1]

#             y = (1 - dynamic_alpha) * y + dynamic_alpha * retrieval_results

#         activity_loss = self.expert_activity_loss(router_prob)

            

#         return y, activity_loss


# Author  : mkw
# Time    : 2025/09/17 10:50:52
# Desc    : ExtremeLSTMMemo
#           - ExpertLSTM encoder/decoder
#           - Standard MoE Head
#           - importance/activity loss + load balancing loss
#           - complete runnable version inside your project

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import DataEmbedding
from layers.att.cross_attention import CrossAttention


# =========================================================
# 1. 基础专家：保持输出维度 = d_model
# =========================================================
class MoEExpert(nn.Module):
    def __init__(self, d_model: int, hidden: Optional[int] = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or (2 * d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        x = self.drop3(self.fc3(x))
        return x


# =========================================================
# 2. Router：基于输入样本统计量进行路由
# =========================================================
class SampleRouterFromX(nn.Module):
    """
    输入:
        x: [B, L, C]
    输出:
        logits: [B, num_experts]
    """
    def __init__(self, c_in: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.c_in = c_in
        in_dim = 3 * c_in  # std + max_abs + last
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=1, unbiased=False)   # [B, C]
        max_abs = x.abs().amax(dim=1)        # [B, C]
        last = x[:, -1, :]                   # [B, C]
        feat = torch.cat([std, max_abs, last], dim=-1)  # [B, 3C]
        return self.net(feat)


# =========================================================
# 3. 标准 MoE Head
#    - 专家输出 d_model
#    - 最后共享映射到 out_dim
# =========================================================
class StandardMoEHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dim: int = 1,
        num_experts: int = 3,
        top_k: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            MoEExpert(d_model=d_model, hidden=2 * d_model, dropout=dropout)
            for _ in range(num_experts)
        ])

        # 所有专家共享的最终输出投影
        self.final_proj = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x: torch.Tensor,              # [B, pred_len, d_model]
        router_probs: torch.Tensor,   # [B, top_k] (已经归一化后的 mix weights)
        topk_experts: torch.Tensor,   # [B, top_k]
    ) -> torch.Tensor:
        B, pred_len, D = x.shape

        moe_out = torch.zeros_like(x)  # [B, pred_len, d_model]

        for k in range(self.top_k):
            expert_idx = topk_experts[:, k]                    # [B]
            weight = router_probs[:, k].view(B, 1, 1)         # [B, 1, 1]

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if not mask.any():
                    continue

                expert_feat = self.experts[e](x[mask])        # [mask_B, pred_len, d_model]
                moe_out[mask] += expert_feat * weight[mask]

        final_out = self.final_proj(moe_out)                  # [B, pred_len, out_dim]
        return final_out


# =========================================================
# 4. 分层 ExpertLSTM
#    - 每层有独立专家池
#    - 每层有独立 router
#    - 返回每层的路由信息，用于 balance loss
# =========================================================
class ExpertLSTM(nn.Module):
    """
    为每一层 LSTM 设计独立的专家分支，支持 top-k 加权融合。
    返回:
        layer_output: [B, L, D]
        (h_n, c_n):   [num_layers, B, D]
        routing_infos: List[Dict]
            每层都包含:
                router_prob  : [B, num_experts]
                topk_experts : [B, top_k]
                mix_weights  : [B, top_k]
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_experts: int,
        top_k_experts: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.dropout = dropout

        # 单层 LSTM 专家；num_layers=1 时内部 dropout 无效，因此这里直接设 0
        self.layer_experts = nn.ModuleList([
            nn.ModuleList([
                nn.LSTM(
                    input_size=d_model,
                    hidden_size=d_model,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0
                ) for _ in range(num_experts)
            ]) for _ in range(num_layers)
        ])

        self.layer_routers = nn.ModuleList([
            SampleRouterFromX(
                c_in=d_model,
                num_experts=num_experts,
                hidden=d_model,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.layer_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[Dict[str, torch.Tensor]]]:

        B, L, D = x.shape
        h_n = torch.zeros(self.num_layers, B, D, device=x.device, dtype=x.dtype)
        c_n = torch.zeros(self.num_layers, B, D, device=x.device, dtype=x.dtype)

        layer_input = x
        routing_infos: List[Dict[str, torch.Tensor]] = []

        for layer_idx in range(self.num_layers):
            # a) 当前层 router
            router_logits = self.layer_routers[layer_idx](layer_input)   # [B, E]
            router_prob = F.softmax(router_logits, dim=-1)               # [B, E]

            # b) top-k 专家
            topk_probs, topk_experts = torch.topk(
                router_prob, k=self.top_k_experts, dim=-1
            )                                                            # [B, K], [B, K]
            mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

            routing_infos.append({
                "router_prob": router_prob,
                "topk_experts": topk_experts,
                "mix_weights": mix_weights,
            })

            # c) 当前层输出容器
            layer_output = torch.zeros_like(layer_input)
            layer_h = torch.zeros(1, B, D, device=x.device, dtype=x.dtype)
            layer_c = torch.zeros(1, B, D, device=x.device, dtype=x.dtype)

            # d) top-k 专家加权融合
            for k in range(self.top_k_experts):
                expert_id = topk_experts[:, k]               # [B]
                token_weight = mix_weights[:, k].view(B, 1, 1)
                state_weight = mix_weights[:, k].view(B, 1)

                for expert_idx in range(self.num_experts):
                    mask = (expert_id == expert_idx)
                    if not mask.any():
                        continue

                    expert_lstm = self.layer_experts[layer_idx][expert_idx]
                    expert_hc = None

                    if hc is not None:
                        expert_h = hc[0][layer_idx:layer_idx + 1, mask, :]   # [1, mask_B, D]
                        expert_c = hc[1][layer_idx:layer_idx + 1, mask, :]
                        expert_hc = (expert_h, expert_c)

                    expert_out, (expert_h, expert_c) = expert_lstm(layer_input[mask], expert_hc)
                    layer_output[mask] += expert_out * token_weight[mask]
                    layer_h[:, mask, :] += expert_h * state_weight[mask].unsqueeze(0)
                    layer_c[:, mask, :] += expert_c * state_weight[mask].unsqueeze(0)

            layer_output = self.layer_dropout(layer_output)

            # e) 写入当前层隐藏状态
            h_n[layer_idx:layer_idx + 1, :, :] = layer_h
            c_n[layer_idx:layer_idx + 1, :, :] = layer_c

            # f) 下一层输入
            layer_input = layer_output

        return layer_output, (h_n, c_n), routing_infos


# =========================================================
# 5. 主模型：ExtremeLSTMMemo
#    - encoder ExpertLSTM
#    - decoder ExpertLSTM
#    - Standard MoE head
#    - importance loss + load loss
# =========================================================
class ExtremeLSTMMemo(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        d_model: int,
        e_layers: int,
        d_layers: int,
        dec_in: int = 5,     # decoder 输入特征维度
        out_dim: int = 1,    # 最终预测维度
        config=None,
    ):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.c_in = c_in
        self.dec_in = dec_in
        self.out_dim = out_dim
        self.dropout = self.config.dropout

        
        

        # -------- experts / retrieval --------
        self.num_experts = 1
        self.retrieval_num = 4
        self.top_k_experts = 1
        self.retrieval_stride = 1

        # -------- loss weights --------
       
        self.importance_loss_weight = 0.1
        self.load_loss_weight = 0.1

        # -------- embedding --------
        self.enc_embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
        self.dec_embedding = DataEmbedding(c_in=dec_in, d_model=d_model, dropout=self.dropout)

        # -------- encoder / decoder --------
        self.encoder = ExpertLSTM(
            d_model=d_model,
            num_layers=e_layers,
            num_experts=self.num_experts,
            top_k_experts=self.top_k_experts,
            dropout=self.dropout
        )
        self.decoder = ExpertLSTM(
            d_model=d_model,
            num_layers=d_layers,
            num_experts=self.num_experts,
            top_k_experts=self.top_k_experts,
            dropout=self.dropout
        )

        self.post_norm = nn.RMSNorm(d_model)
        self.xattn = CrossAttention(d_model)

        # -------- head router --------
        self.router = SampleRouterFromX(
            c_in=c_in,
            num_experts=self.num_experts,
            hidden=d_model,
            dropout=self.dropout
        )

        # -------- Standard MoE Head --------
        self.moe_head = StandardMoEHead(
            d_model=d_model,
            out_dim=out_dim,
            num_experts=self.num_experts,
            top_k=self.top_k_experts,
            dropout=self.dropout
        )

        self.fuse_proj = nn.Linear(2 * d_model, d_model)

        # -------- retrieval memory --------
        self.keys = None      # [N, seq_len, c_in]
        self.values = None    # [N, pred_len, out_dim]
        self.index = 0

        # 可选：用于调试查看每次 forward 的明细
        self.latest_aux_dict = {}

    # -----------------------------------------------------
    # device helper
    # -----------------------------------------------------
    @property
    def runtime_device(self):
        return next(self.parameters()).device

    # -----------------------------------------------------
    # 6. balance loss 核心：importance + load
    # -----------------------------------------------------
    def compute_router_balance_losses(
        self,
        router_prob: torch.Tensor,    # [B, E]
        topk_experts: torch.Tensor,   # [B, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            importance_loss: soft routing 概率分布均衡损失
            load_loss      : hard top-k 选择频率均衡损失
        """
        num_experts = router_prob.size(-1)

        # ---------- importance / activity ----------
        # 每个专家在 batch 内被分配到的平均概率
        importance = router_prob.mean(dim=0)   # [E]
        importance = importance / (importance.sum() + 1e-8)

        target = torch.full_like(importance, 1.0 / num_experts)
        importance_loss = F.mse_loss(importance, target)

        # ---------- load balancing ----------
        # 基于 top-k 硬选择统计每个专家真实被选中的频率
        # topk_experts: [B, K]
        expert_load = F.one_hot(topk_experts, num_classes=num_experts).float()  # [B, K, E]
        expert_load = expert_load.sum(dim=1)                                     # [B, E]
        load = expert_load.mean(dim=0)                                           # [E]
        load = load / (load.sum() + 1e-8)                                        # 归一化成概率分布

        load_loss = F.mse_loss(load, target)

        return importance_loss, load_loss

    def aggregate_balance_loss(
        self,
        head_router_prob: torch.Tensor,
        head_topk_experts: torch.Tensor,
        enc_routing_infos: List[Dict[str, torch.Tensor]],
        dec_routing_infos: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        聚合 head / encoder 每层 / decoder 每层 的两种损失
        """
        importance_losses = []
        load_losses = []

        # 1) head
        head_imp_loss, head_load_loss = self.compute_router_balance_losses(
            head_router_prob, head_topk_experts
        )
        importance_losses.append(head_imp_loss)
        load_losses.append(head_load_loss)

        # 2) encoder 每层
        enc_imp_each = []
        enc_load_each = []
        for info in enc_routing_infos:
            imp_loss, load_loss = self.compute_router_balance_losses(
                info["router_prob"], info["topk_experts"]
            )
            importance_losses.append(imp_loss)
            load_losses.append(load_loss)
            enc_imp_each.append(imp_loss)
            enc_load_each.append(load_loss)

        # 3) decoder 每层
        dec_imp_each = []
        dec_load_each = []
        for info in dec_routing_infos:
            imp_loss, load_loss = self.compute_router_balance_losses(
                info["router_prob"], info["topk_experts"]
            )
            importance_losses.append(imp_loss)
            load_losses.append(load_loss)
            dec_imp_each.append(imp_loss)
            dec_load_each.append(load_loss)

        total_importance_loss = torch.stack(importance_losses).mean()
        total_load_loss = torch.stack(load_losses).mean()

        balance_loss = (
            self.importance_loss_weight * total_importance_loss +
            self.load_loss_weight * total_load_loss
        )

        aux_dict = {
            "balance_loss": balance_loss.detach(),
            "total_importance_loss": total_importance_loss.detach(),
            "total_load_loss": total_load_loss.detach(),
            "head_importance_loss": head_imp_loss.detach(),
            "head_load_loss": head_load_loss.detach(),
            "enc_importance_loss": torch.stack(enc_imp_each).mean().detach() if len(enc_imp_each) > 0 else torch.tensor(0.0, device=self.runtime_device),
            "enc_load_loss": torch.stack(enc_load_each).mean().detach() if len(enc_load_each) > 0 else torch.tensor(0.0, device=self.runtime_device),
            "dec_importance_loss": torch.stack(dec_imp_each).mean().detach() if len(dec_imp_each) > 0 else torch.tensor(0.0, device=self.runtime_device),
            "dec_load_loss": torch.stack(dec_load_each).mean().detach() if len(dec_load_each) > 0 else torch.tensor(0.0, device=self.runtime_device),
        }

        return balance_loss, aux_dict

    # -----------------------------------------------------
    # 7. retrieval memory
    # -----------------------------------------------------
    def construct_index(self, num: int):
        """
        构建检索库
        keys  : [num, seq_len, c_in]
        values: [num, pred_len, out_dim]
        """
        device = self.runtime_device
        self.keys = torch.zeros(num, self.seq_len, self.c_in, device=device)
        self.values = torch.zeros(num, self.pred_len, self.dec_in, device=device)
        self.index = 0

    @torch.no_grad()
    def add_key_value(self, x_enc: torch.Tensor, y: torch.Tensor, index: torch.Tensor):
        """
        x_enc : [B, seq_len, c_in]
        y     : [B, pred_len, out_dim]
        index : [B] or slice-like indices
        """
        if self.keys is None or self.values is None:
            raise RuntimeError("Please call construct_index(num) before add_key_value().")

        self.keys[index, :, :] = x_enc
        self.values[index, :, :] = y
        self.index += x_enc.size(0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cosine_similarity(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        queries: [B, L, C] or [B, D]
        keys   : [N, L, C] or [N, D]
        return : [B, N]
        """
        if len(queries.shape) == 3:
            B = queries.size(0)
            N = keys.size(0)
            queries = queries.reshape(B, -1)
            keys = keys.reshape(N, -1)

            q_norm = F.normalize(queries, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())

        elif len(queries.shape) == 2:
            q_norm = F.normalize(queries, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())

        else:
            raise ValueError(f"Unsupported query shape: {queries.shape}")

    def retrieval(self, x: torch.Tensor, index: Optional[torch.Tensor]):
        """
        x    : [B, seq_len, c_in]
        return:
            output: [B, pred_len, out_dim]
            sims  : [B, k]
            t     : dummy
        """
        bs = x.shape[0]
        k = min(self.retrieval_num, self.index)

        queries = x
        keys = self.keys[:self.index]
        values = self.values[:self.index]

        dis = self.cosine_similarity(queries, keys)  # [B, N]

        # 训练时屏蔽自身附近样本
        if self.training and index is not None:
            self_range = torch.arange(
                -self.seq_len, self.seq_len + 1, device=x.device
            ).unsqueeze(0)  # [1, 2*seq_len+1]

            invalid_index = index.unsqueeze(1) + self_range
            invalid_index = invalid_index // self.retrieval_stride
            invalid_index[invalid_index < 0] = 0
            invalid_index[invalid_index >= self.index] = self.index - 1

            row_idx = torch.arange(bs, device=x.device).unsqueeze(1).repeat(1, invalid_index.size(1))
            dis[row_idx, invalid_index] = -100.0

        dis_topk, indices_topk = torch.topk(dis, dim=1, k=k)                  # [B, k]
        sims = dis_topk
        probs_topk = torch.softmax(dis_topk, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, k, 1, 1]

        retrieved_values = values[indices_topk]                               # [B, k, pred_len, out_dim]
        output = torch.sum(probs_topk * retrieved_values, dim=1)              # [B, pred_len, out_dim]

        return output, sims, 0

    # -----------------------------------------------------
    # 8. forward
    # -----------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        dec_input: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        mode: str = "train",
        return_aux: bool = False,
    ):
        B = x.size(0)

        # ---------------- head router ----------------
        head_router_logits = self.router(x)                    # [B, E]
        head_router_prob = torch.softmax(head_router_logits, dim=-1)

        head_topk_probs, head_topk_experts = torch.topk(
            head_router_prob, k=self.top_k_experts, dim=-1
        )
        head_mix_weights = head_topk_probs / (
            head_topk_probs.sum(dim=-1, keepdim=True) + 1e-8
        )                                                     # [B, K]

        # ---------------- embedding ----------------
        x_emb_hist = self.enc_embedding(x)                    # [B, seq_len, d_model]

        
        dec_emb = self.dec_embedding(dec_input)               # [B, pred_len, d_model]

        # ---------------- encoder / decoder ----------------
        enc_out, (h_n, c_n), enc_routing_infos = self.encoder(x_emb_hist)
        dec_out, _, dec_routing_infos = self.decoder(dec_emb, (h_n, c_n))
        dec_out = dec_out[:, -self.pred_len:, :]

        # ---------------- cross attention + fuse ----------------
        ctx, _ = self.xattn(dec_out, enc_out, enc_out)        # [B, pred_len, d_model]
        fused = torch.cat([dec_out, ctx], dim=-1)             # [B, pred_len, 2*d_model]
        fused = self.fuse_proj(fused)                         # [B, pred_len, d_model]
        final_shared = self.post_norm(fused)                  # [B, pred_len, d_model]

        # ---------------- Standard MoE Head ----------------
        y = self.moe_head(final_shared, head_mix_weights, head_topk_experts)  # [B, pred_len, out_dim]

        # ---------------- test-time retrieval fusion ----------------
        if mode == "test":
            retrieval_results, sims, _ = self.retrieval(x, sample_ids)

            sim_mean = torch.mean(sims, dim=-1, keepdim=True)                 # [B, 1]
            dynamic_alpha = 0.05 * (
                (sim_mean - sim_mean.min()) /
                (sim_mean.max() - sim_mean.min() + 1e-8)
            )
            dynamic_alpha = dynamic_alpha.unsqueeze(-1)                       # [B, 1, 1]

            # 两者现在维度一致：[B, pred_len, out_dim]
            y = (1 - dynamic_alpha) * y + dynamic_alpha * retrieval_results

        # ---------------- balance loss ----------------
        balance_loss, aux_dict = self.aggregate_balance_loss(
            head_router_prob=head_router_prob,
            head_topk_experts=head_topk_experts,
            enc_routing_infos=enc_routing_infos,
            dec_routing_infos=dec_routing_infos,
        )

        self.latest_aux_dict = aux_dict

        if return_aux:
            return y, balance_loss, aux_dict
        return y, balance_loss