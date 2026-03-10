#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   ExtremeLSTMMemo (MoE on LSTM Layer + Standard MoE Head)

from layers.embedding import DataEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.att.cross_attention import CrossAttention

class MoEExpert(nn.Module):
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or (2 * d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        x = self.drop3(self.act(self.fc3(x)))
        return x

class StandardMoEHead(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 3, top_k: int = 2, dropout: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 1. 专家池：每个专家输出维度=d_model
        self.experts = nn.ModuleList([
            MoEExpert(d_model, hidden=2*d_model, dropout=dropout) 
            for _ in range(num_experts)
        ])
        
        # 2. 最后统一映射到1维（所有专家共享）
        self.final_proj = nn.Linear(d_model, 1)

    def forward(self, x, router_probs, topk_experts):
        
        B, pred_len, D = x.shape
        
        # 初始化MoE输出（维度保持d_model）
        moe_out = torch.zeros_like(x)  # [B, pred_len, d_model]
        
        # 遍历top-k专家，按索引选择并加权融合
        for k in range(self.top_k):
            # 获取第k个专家的索引和权重
            expert_idx = topk_experts[:, k]  # [B]
            weight = router_probs[:, k].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            
            # 按索引选择对应专家计算
            for e in range(self.num_experts):
                # 找到选中当前专家的样本掩码
                mask = (expert_idx == e)
                if not mask.any():
                    continue
                
                # 专家前向计算（输出d_model） + 加权累加
                expert_feat = self.experts[e](x[mask])  # [mask_B, pred_len, d_model]
                moe_out[mask] += expert_feat * weight[mask]
        
        # 最后统一映射到1维
        final_out = self.final_proj(moe_out)  # [B, pred_len, 1]
        return final_out
    
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
        std = x.std(dim=1, unbiased=False)  # [B, C]
        max_abs = x.abs().amax(dim=1)  # [B, C]
        last = x[:, -1, :]  # [B, C]
        feat = torch.cat([std, max_abs, last], dim=-1)  # [B, 3C]
        return self.net(feat)

# 分层专家LSTM（保持不变）
class ExpertLSTM(nn.Module):
    """为每一层LSTM设计独立的专家分支，支持top-k加权融合"""
    def __init__(self, d_model: int, num_layers: int, num_experts: int, 
                 top_k_experts: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.dropout = dropout

        # 1. 为每一层LSTM创建多个专家（单层LSTM）
        self.layer_experts = nn.ModuleList([
            nn.ModuleList([
                nn.LSTM(
                    input_size=d_model,
                    hidden_size=d_model,
                    num_layers=1,
                    batch_first=True,
                    dropout=dropout
                ) for _ in range(num_experts)
            ]) for _ in range(num_layers)
        ])

        # 2. 每层独立的路由器
        self.layer_routers = nn.ModuleList([
            SampleRouterFromX(
                c_in=d_model, 
                num_experts=num_experts, 
                hidden=d_model, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, hc=None):
        B, L, D = x.shape
        h_n = torch.zeros(self.num_layers, B, D, device=x.device)
        c_n = torch.zeros(self.num_layers, B, D, device=x.device)

        layer_input = x
        for layer_idx in range(self.num_layers):
            # a. 当前层的路由器：计算专家权重 [B, num_experts]
            router_logits = self.layer_routers[layer_idx](layer_input)
            router_prob = F.softmax(router_logits, dim=-1)

            # b. 选择top-k专家并归一化权重
            topk_probs, topk_experts = torch.topk(router_prob, k=self.top_k_experts, dim=-1)
            mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # [B, k]

            # c. 初始化当前层输出
            layer_output = torch.zeros_like(layer_input)
            layer_h = torch.zeros(1, B, D, device=x.device)
            layer_c = torch.zeros(1, B, D, device=x.device)

            # d. 遍历top-k专家，加权融合输出
            for k in range(self.top_k_experts):
                expert_id = topk_experts[:, k]
                weight = mix_weights[:, k].unsqueeze(-1).unsqueeze(-1)

                for expert_idx in range(self.num_experts):
                    mask = (expert_id == expert_idx)
                    if not mask.any():
                        continue

                    expert_lstm = self.layer_experts[layer_idx][expert_idx]
                    expert_hc = None
                    if hc is not None:
                        expert_h = hc[0][layer_idx:layer_idx+1, mask, :]
                        expert_c = hc[1][layer_idx:layer_idx+1, mask, :]
                        expert_hc = (expert_h, expert_c)

                    expert_out, (expert_h, expert_c) = expert_lstm(layer_input[mask], expert_hc)
                    layer_output[mask] += expert_out * weight[mask]
                    layer_h[:, mask, :] += expert_h * weight[mask].squeeze(1)
                    layer_c[:, mask, :] += expert_c * weight[mask].squeeze(1)

            # e. 更新当前层的隐藏状态
            h_n[layer_idx:layer_idx+1, :, :] = layer_h
            c_n[layer_idx:layer_idx+1, :, :] = layer_c

            # f. 作为下一层的输入
            layer_input = layer_output

        return layer_output, (h_n, c_n)

# ========== 主模型：替换为标准MoE Head ==========
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
        self.retrieval_num = 8
        self.top_k_experts = 2  # MoE统一用top-2
        self.retrieval_stride = 1
        
        # -------- Embedding + pred tokens --------
        self.enc_embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
        self.dec_embedding = DataEmbedding(c_in=5, d_model=d_model, dropout=self.dropout)

        # -------- 专家LSTM编码器/解码器 --------
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

        # -------- router --------
        router_hidden = self.d_model
        router_dropout = self.dropout
        self.router = SampleRouterFromX(c_in=c_in, num_experts=self.num_experts, hidden=router_hidden, dropout=router_dropout)

        # -------- 核心修正：标准MoE Head --------
        self.moe_head = StandardMoEHead(
            d_model=d_model,
            num_experts=self.num_experts,
            top_k=self.top_k_experts,
            dropout=self.dropout
        )
        
        self.fuse_proj = nn.Linear(2 * d_model, d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def construct_index(self, num):
        key_len = self.seq_len
        self.keys = torch.zeros(num, key_len, 1, device=self.device)
        self.values = torch.zeros(num, self.pred_len, 1, device=self.device)
        self.index = 0

    @torch.no_grad()
    def add_key_value(self, x_enc, y, index):
        bs = x_enc.shape[0]
        x_key = x_enc
        self.keys[index, :, :] = x_key
        self.values[index, :, :] = y
        self.index += bs
        if torch.cuda.is_available():
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
        
    def forward(self, x, x_mark=None, dec_input=None, sample_ids=None, mode='train'):
        B = x.size(0)
        # ---------------- routing ----------------
        router_logits = self.router(x)                       # [B, num_experts]
        router_prob = torch.softmax(router_logits, dim=-1)    # [B, num_experts]

        # ---------------- embedding ----------------
        x_emb_hist = self.enc_embedding(x)                        # [B, seq_len, d_model]

        # =========================================================
        # LSTM backbone: 专家LSTM编码器 + 专家LSTM解码器
        # =========================================================
        enc_out, (h_n, c_n) = self.encoder(x_emb_hist)              # [B, seq_len, d_model]
        dec_input = self.dec_embedding(dec_input)
        dec_out, _ = self.decoder(dec_input, (h_n, c_n))             # [B, pred_len, d_model]
        dec_out = dec_out[:, -self.pred_len:, :]
        
        # 交叉注意力 + 特征融合（输出仍为d_model）
        ctx, _ = self.xattn(dec_out, enc_out, enc_out)
        fused = torch.cat([dec_out, ctx], dim=-1)                  # [B, pred_len, 2*d_model]
        fused = self.fuse_proj(fused)                              # [B, pred_len, d_model]
        final_shared = self.post_norm(fused)                        # [B, pred_len, d_model]
        # y = self.out_proj(final_shared)

        # =========================================================
        # 核心修正：标准MoE Head计算
        # =========================================================
        # 1. 选择top-k专家并归一化权重
        topk_result = torch.topk(router_prob, k=self.top_k_experts, dim=-1)
        topk_probs = topk_result.values     # [B, top_k]
        topk_experts = topk_result.indices  # [B, top_k]
        mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # [B, top_k]

        # 2. 标准MoE前向：专家特征变换（d_model） + 最后映射到1维
        y = self.moe_head(final_shared, mix_weights, topk_experts)  # [B, pred_len, 1]

        # if mode == 'test':
        #     print("Retrieval in test mode...........")
        #     retrieval_results, sims, t = self.retrieval(x, sample_ids) # 检索得到的结果
        #     print("Retrieval end...........")
           
        #     sim_mean = torch.mean(sims, dim=-1).unsqueeze(-1)  # [bs, 1, 1]
        #     dynamic_alpha = 0.05 * (sim_mean - sim_mean.min()) / (sim_mean.max() - sim_mean.min() + 1e-8)
        #     y = (1 - dynamic_alpha) * y + dynamic_alpha * retrieval_results

        return y 

