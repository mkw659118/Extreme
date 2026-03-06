#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   ExtremeLSTMMemo
from time import time

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

class ExtremeLSTMMemo(nn.Module):
    def __init__(
        self,
        c_in: int ,
        seq_len: int,
        pred_len: int,                
        d_model: int,
        e_layers: int, 
        d_layers: int, 
        config=None,
    ):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.c_in = c_in
        self.dropout = self.config.dropout
        self.device = self.config.device

        # -------- expert definition --------
        self.num_experts = 3
        self.retrieval_num = 10
        self.alpha = 0.5
        self.top_k_experts = 2
        self.retrieval_stride = 1
        
        # -------- Embedding + pred tokens --------

        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

        enc_layers = int(e_layers)
        dec_layers = int(d_layers)

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

        # -------- router --------
        router_hidden = self.d_model
        router_dropout = self.dropout
        self.router = SampleRouterFromX(c_in=c_in, num_experts=self.num_experts, hidden=router_hidden, dropout=router_dropout)

        # -------- heads --------
        # self.expert_heads = nn.ModuleList([
        #     NormalHead(d_model),
        #     MidHead(d_model, hidden=d_model, dropout=self.dropout),
        #     ExtremeHead(d_model, hidden=2 * d_model, dropout=self.dropout),
        # ])

        self.expert_heads = nn.ModuleList([
            ExtremeHead(d_model, hidden=2 * d_model, dropout=self.dropout),
            ExtremeHead(d_model, hidden=2 * d_model, dropout=self.dropout),
            ExtremeHead(d_model, hidden=2 * d_model, dropout=self.dropout)
        ])
        
        self.fuse_proj = nn.Linear(2 * d_model, d_model)

    def construct_index(self, num):
        key_len = self.seq_len
        self.keys = torch.zeros(num, key_len, 1, device=self.device)
        self.values = torch.zeros(num, self.pred_len, 1, device=self.device)
        self.index = 0

    @torch.no_grad()
    def add_key_value(self, x_enc, y, index):
        bs = x_enc.shape[0]
        x_key = x_enc
        # key是输入序列
        self.keys[index, :, :] = x_key
        # value是真实值
        self.values[index, :, :] = y
        # 给每一条序列一个唯一的索引，方便后续检索
        self.index += bs
        torch.cuda.empty_cache()


    def retrieval(self, x, index):
        bs = x.shape[0]
        k = self.retrieval_num
        queries = x[..., 0].unsqueeze(-1)
        keys = self.keys
        # keys = self.keys.transpose(2, 1).reshape(-1, self.seq_len)
        dis = self.cosine_similarity(queries, keys)
        # 检索时，禁止检索到自己和附近的序列（因为它们的标签很可能重叠，容易导致过拟合），这里的距离是以index为基准的绝对距离
        if self.training:
            # offline
            self_range = torch.arange(-self.seq_len, self.seq_len + 1, device=x.device).unsqueeze(0)
            invalid_index = index.unsqueeze(1) + self_range
            invalid_index = invalid_index // self.retrieval_stride
            invalid_index[torch.where(invalid_index < 0)] = 0
            invalid_index[torch.where(invalid_index >= self.index)] = self.index - 1
            row_idx = torch.arange(x.shape[0]).unsqueeze(1).repeat(1, 2 * self.seq_len + 1)
            dis[:, row_idx, invalid_index] = -100

            # online
            # invalid_index = torch.arange(self.index).unsqueeze(0).repeat(bs,1) #bs*len
            # index = index // self.retrieval_stride
            # for i in range(bs):
            #     mask_index = min(max(k, index[i]),self.index - 1)
            #     invalid_index[i, :mask_index] = mask_index
            # row_idx = torch.arange(x.shape[0]).unsqueeze(1).repeat(1, self.index)
            # dis[:, row_idx, invalid_index] = 0
        dis_topk, indices_topk = torch.topk(dis, dim=2, k=k)
        sims = dis_topk.permute(1, 0, 2) # bs*c*k
        probs_topk = torch.softmax(dis_topk, dim=2).unsqueeze(-1)  # c*bs*k*1
        

        # values = self.values.permute(2, 0, 1)[torch.arange(self.in_c).unsqueeze(1).repeat(1, bs * k),
        #          indices_topk.view(self.in_c, -1), :]
        # values = values.reshape(self.in_c, bs, -1, self.pred_len)
        values = self.value_permute  # [in_c, N, pred_len]

        # reshape 为 [1, in_c, N, pred_len]，为 batch gather 做准备
        values = values.unsqueeze(0)  # [1, in_c, N, pred_len]

        # indices_topk.shape = [bs, in_c, k]
        # 需要扩展为 [bs, in_c, k, 1] 以便 gather
        indices = indices_topk.permute(1, 0, 2).unsqueeze(-1)  # [in_c, bs, k, 1]

        # 转换 values 为 [in_c, 1, N, pred_len] 以与 indices 对齐
        values = values.expand(bs, -1, -1, -1)  # [in_c, 1, N, pred_len]

        # gather
        values = torch.gather(values, 2, indices.expand(-1, -1, -1, values.size(-1))).permute(1,0,2,3)  # [in_c, bs, k, pred_len]

        output = torch.sum(probs_topk * values, dim=2).permute(1, 2, 0)  # weighted-sum ver
        return output, sims, 0

    def cosine_similarity(self, queries, keys):
        # equals to person_similarity when revin=True, since std=1, mean=0
        if len(queries.shape) == 2:  # B*L
            q_norm = torch.nn.functional.normalize(queries, p=2, dim=-1)
            k_norm = torch.nn.functional.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())
        elif len(queries.shape) == 3:  # B*L*C
            queries = queries.permute(2, 0, 1)
            keys = keys.permute(2, 0, 1)
            q_norm = torch.nn.functional.normalize(queries, p=2, dim=-1)
            k_norm = torch.nn.functional.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.permute(0, 2, 1))
        
    def forward(self, x, x_mark=None, sample_ids=None, mode='train'):
        
        B = x.size(0)
        # ---------------- routing ----------------
        router_logits = self.router(x)                       # [B, E]
        router_prob = torch.softmax(router_logits, dim=-1)    # [B, E]

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

        # 2) 路由选择：对每个样本，从 router_prob 中选出概率最大的 top-k 个专家
        k = self.top_k_experts
        topk_result = torch.topk(router_prob, k=k, dim=-1)

        topk_probs = topk_result.values     # [B, k]，top-k 专家的路由概率（尚未在 top-k 内归一化）
        topk_experts = topk_result.indices  # [B, k]，top-k 专家的编号（expert id）

        # 3) 权重归一化：只在 top-k 范围内做归一化，使每个样本的 top-k 权重和为 1
        mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # [B, k]

        # 4) 收集对应专家输出：把每个样本选中的 top-k 专家输出从 expert_preds 中 gather 出来
        # chosen_expert_preds: [B, pred_len, k]
        expert_index = topk_experts[:, None, :].expand(B, self.pred_len, k)       # [B, pred_len, k]，把索引扩展到每个预测步
        chosen_expert_preds = expert_preds.gather(dim=-1, index=expert_index)     # [B, pred_len, k]，取出 top-k 专家的预测

        # 5) 加权融合：用 top-k 权重对对应专家输出加权求和，得到最终预测
        mix_weights = mix_weights[:, None, :].expand(B, self.pred_len, k)         # [B, pred_len, k]，把权重扩展到每个预测步
       
        y= (chosen_expert_preds * mix_weights).sum(dim=-1, keepdim=True)         # [B, pred_len, 1]，最终预测

        if mode == 'test':
            print("Retrieval in test mode...........")
            retrieval_results, sims, t = self.retrieval(x, sample_ids) # 检索得到的结果
            print("Retrieval end...........")
            
            
            sim_mean = torch.mean(sims, dim=-1).unsqueeze(-1)  # [bs, 1, 1]
            # 归一化到[0, 0.2]区间（避免权重过大），替代固定0.1
            dynamic_alpha = 0.2 * (sim_mean - sim_mean.min()) / (sim_mean.max() - sim_mean.min() + 1e-8)
            y = (1 - dynamic_alpha) * y + dynamic_alpha * retrieval_results

        return y 