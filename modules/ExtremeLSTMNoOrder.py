#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   ExtremeLSTM
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from layers.embedding import DataEmbedding
# 残差记忆库，全存，容量大小1024

class NormalHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):  # [B, pred_len, d_model]
        return self.proj(x)


class MidHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or d_model
        self.fc = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, 1)   # 关键：保留 proj

    def forward(self, x):
        x = self.drop(self.act(self.fc(x)))
        return self.proj(x)


class ExtremeHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or (2 * d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, d_model)     # 压回 d_model
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, 1)         # 所有专家统一 proj: d_model -> 1

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        x = self.drop3(self.act(self.fc3(x)))
        return self.proj(x)


class SampleRouterFromX(nn.Module):
   
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

    def forward(self, x):
        std = x.std(dim=1, unbiased=False)  # 1) std: [B, C]
        max_abs = x.abs().amax(dim=1)  # 2) max_abs: [B, C]
        last = x[:, -1, :]  # 3) last: [B, C]
        feat = torch.cat([std, max_abs, last], dim=-1)  # [B, 3C]
        return self.net(feat)
    
class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V):
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # [B,P,T]
        attn = F.softmax(scores, dim=-1)
        ctx  = torch.matmul(attn, v)  # [B,P,H]
        return ctx, attn  

# class ExtremeLSTMNoOrder(nn.Module):
#     def __init__(
#         self,
#         seq_len: int,
#         pred_len: int,
#         patch_len: int,              
#         d_model: int,
#         win_size: int,               
#         revin: bool,
#         num_heads: int,              
#         use_memory: bool,
#         num_layers_intra_patch: int, 
#         num_layers_inter_patch: int, 
#         config=None,
#         c_in: int = 10,
#     ):
#         super().__init__()
#         self.config = config
#         self.revin = revin
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.d_model = d_model
#         self.c_in = c_in
#         self.dropout = self.config.dropout

#         # -------- expert definition --------
#         self.num_experts = 3
        
#         # -------- Embedding + pred tokens --------

#         self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
#         self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

#         enc_layers = int(num_layers_intra_patch)
#         dec_layers = int(num_layers_inter_patch)

#         self.encoder = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=enc_layers,
#             batch_first=True,
#             dropout=self.dropout,
#         )
#         self.decoder = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=dec_layers,
#             batch_first=True,
#             dropout=self.dropout,
#         )

#         self.post_norm = nn.RMSNorm(d_model)
#         self.xattn = CrossAttention(d_model)

#         # -------- router --------
#         router_hidden = self.d_model
#         router_dropout = self.dropout
#         self.router = SampleRouterFromX(c_in=c_in, num_experts=self.num_experts, hidden=router_hidden, dropout=router_dropout)

#         # -------- heads --------
#         self.expert_heads = nn.ModuleList([
#             NormalHead(d_model),
#             MidHead(d_model, hidden=d_model, dropout=self.dropout),
#             ExtremeHead(d_model, hidden=2 * d_model, dropout=self.dropout),
#         ])
        
#         self.fuse_proj = nn.Linear(2 * d_model, d_model)


#         # -------- top-k gating --------
#         self.top_k = 2
        
#         # -------- GMM label slices --------
#         self.gmm_pt_start  = 2
#         self.gmm_pt_end    = 5
#         self.gmm_seq_start = 7
#         self.gmm_seq_end   = 10

#     def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
#         """
#         x: [B, seq_len, c_in]
#         return y: [B, pred_len, 1]
#         """
#         B = x.size(0)
#         # ---------------- routing ----------------
#         router_logits = self.router(x)                       # [B, E]
#         router_prob = torch.softmax(router_logits, dim=-1)    # [B, E]

#         # ---------------- embedding ----------------
#         x_emb_hist = self.embedding(x)                        # [B, seq_len, d_model]

#         # =========================================================
#         # LSTM backbone: encoder history -> decoder pred_tokens
#         # =========================================================
#         enc_out, (h_n, c_n) = self.encoder(x_emb_hist)              # h_n/c_n: [enc_layers, B, d_model]

#         pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, d_model]
#         dec_out, _ = self.decoder(pred_token, (h_n, c_n))             # [B, pred_len, d_model]
#         ctx, _ = self.xattn(dec_out, enc_out, enc_out)
#         fused = torch.cat([dec_out, ctx], dim=-1)
#         fused = self.fuse_proj(fused)          # [B, pred_len, d_model]

#         final_shared = self.post_norm(fused)                        # [B, pred_len, d_model]

#         # 1) 专家头：计算每个 expert head 的输出并在最后一维拼接
#         # expert_preds: [B, pred_len, E]，E 为专家数
#         expert_preds = torch.cat([head(final_shared) for head in self.expert_heads], dim=-1)

#         # 2) 路由选择：对每个样本，从 router_prob 中选出概率最大的 top-k 个专家
#         k = self.top_k
#         topk_result = torch.topk(router_prob, k=k, dim=-1)

#         topk_probs = topk_result.values     # [B, k]，top-k 专家的路由概率（尚未在 top-k 内归一化）
#         topk_experts = topk_result.indices  # [B, k]，top-k 专家的编号（expert id）

#         # 3) 权重归一化：只在 top-k 范围内做归一化，使每个样本的 top-k 权重和为 1
#         mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # [B, k]

#         # 4) 收集对应专家输出：把每个样本选中的 top-k 专家输出从 expert_preds 中 gather 出来
#         # chosen_expert_preds: [B, pred_len, k]
#         expert_index = topk_experts[:, None, :].expand(B, self.pred_len, k)       # [B, pred_len, k]，把索引扩展到每个预测步
#         chosen_expert_preds = expert_preds.gather(dim=-1, index=expert_index)     # [B, pred_len, k]，取出 top-k 专家的预测

#         # 5) 加权融合：用 top-k 权重对对应专家输出加权求和，得到最终预测
#         mix_weights = mix_weights[:, None, :].expand(B, self.pred_len, k)         # [B, pred_len, k]，把权重扩展到每个预测步
#         y = (chosen_expert_preds * mix_weights).sum(dim=-1, keepdim=True)         # [B, pred_len, 1]，最终预测

#         return y 
    
class ExtremeLSTMNoOrder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,              
        d_model: int,
        win_size: int,                # 滑动窗口大小
        revin: bool,
        num_heads: int,               # 自注意力头数
        use_memory: bool,
        num_layers_intra_patch: int, 
        num_layers_inter_patch: int, 
        config=None,
        c_in: int = 10,              # 输入特征数
    ):
        super().__init__()
        self.config = config
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.c_in = c_in
        self.dropout = self.config.dropout

        # -------- expert definition --------
        self.num_experts = 3
        
        # -------- Embedding + pred tokens --------
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

        enc_layers = int(num_layers_intra_patch)
        dec_layers = int(num_layers_inter_patch)

        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=enc_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=dec_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        self.post_norm = nn.RMSNorm(d_model)
        self.xattn = CrossAttention(d_model)

        # -------- heads --------
        self.expert_heads = nn.ModuleList([
            NormalHead(d_model),
            MidHead(d_model, hidden=d_model, dropout=self.dropout),
            ExtremeHead(d_model, hidden=2 * d_model, dropout=self.dropout),
        ])
        
        self.fuse_proj = nn.Linear(2 * d_model, d_model)
        
        # -------- GMM label slices --------
        self.gmm_pt_start  = 2
        self.gmm_pt_end    = 5
        self.gmm_seq_start = 7
        self.gmm_seq_end   = 10


    def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
        """
        x: [B, seq_len, c_in]
        return y: [B, pred_len, 1]
        """
        B = x.size(0)

        # ---------------- embedding ----------------
        x_emb_hist = self.embedding(x)                        # [B, seq_len, d_model]

        # =========================================================
        # LSTM backbone: encoder history -> decoder pred_tokens
        # =========================================================
        enc_out, (h_n, c_n) = self.encoder(x_emb_hist)              # h_n/c_n: [enc_layers, B, d_model]

        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, d_model]
        dec_out, _ = self.decoder(pred_token, (h_n, c_n))             # [B, pred_len, d_model]
        ctx, _ = self.xattn(dec_out, enc_out, enc_out)
        fused = torch.cat([dec_out, ctx], dim=-1)
        fused = self.fuse_proj(fused)          # [B, pred_len, d_model]

        final_shared = self.post_norm(fused)                        # [B, pred_len, d_model]

        # 1) 专家头：计算每个 expert head 的输出并在最后一维拼接
        # expert_preds: [B, pred_len, E]，E 为专家数
        expert_preds = torch.cat([head(final_shared) for head in self.expert_heads], dim=-1)

        # 2) 计算点级和序列级概率
        point_prob = x[:,:, self.gmm_pt_start:self.gmm_pt_end]  
        seq_prob = x[:,:, self.gmm_seq_start:self.gmm_seq_end]

        # 3) 合并点级和序列级概率
        combined_prob = point_prob + seq_prob  # [B, E]，加权概率

        # 4) 选择最大概率对应的专家（生成伪标签）
        expert_choice = torch.argmax(combined_prob, dim=-1)  # [B]，每个样本选择最大概率对应的专家

        # 5) 根据选择的专家选择对应的预测
        chosen_expert_preds = expert_preds.gather(dim=-1, index=expert_choice.unsqueeze(-1).expand(B, self.pred_len, 1))  # [B, pred_len, 1]

        return chosen_expert_preds