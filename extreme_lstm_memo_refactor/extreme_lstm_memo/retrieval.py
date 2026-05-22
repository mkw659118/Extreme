from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class RetrievalMemory:
    """
    Tensor-based retrieval memory for storing historical input-output pairs.

    This class is intentionally not an nn.Module because it stores an external
    non-parametric memory bank rather than learnable parameters.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        value_dim: int,
        device: torch.device,
        stride: int = 1,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.value_dim = value_dim
        self.device = device
        self.stride = stride
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.index = 0

    def construct_index(self, num: int) -> None:
        self.keys = torch.zeros(num, self.seq_len, self.c_in, device=self.device)
        self.values = torch.zeros(num, self.pred_len, self.value_dim, device=self.device)
        self.index = 0

    @torch.no_grad()
    def add_key_value(self, x_enc: torch.Tensor, y: torch.Tensor, index: torch.Tensor) -> None:
        if self.keys is None or self.values is None:
            raise RuntimeError("Retrieval memory has not been constructed yet.")
        self.keys[index, :, :] = x_enc
        self.values[index, :, :] = y
        self.index += x_enc.size(0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def cosine_similarity(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if len(queries.shape) == 3:
            batch_size = queries.size(0)
            num_keys = keys.size(0)
            queries = queries.reshape(batch_size, -1)
            keys = keys.reshape(num_keys, -1)
        elif len(queries.shape) != 2:
            raise ValueError(f"Unsupported query shape: {queries.shape}")

        q_norm = F.normalize(queries, p=2, dim=-1)
        k_norm = F.normalize(keys, p=2, dim=-1)
        return torch.matmul(q_norm, k_norm.t())

    def query(
        self,
        x: torch.Tensor,
        sample_ids: Optional[torch.Tensor],
        top_k: int,
        exclude_self_window: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.keys is None or self.values is None or self.index == 0:
            raise RuntimeError("Retrieval index has not been constructed yet.")

        batch_size = x.shape[0]
        k = min(top_k, self.index)
        keys = self.keys[: self.index]
        values = self.values[: self.index]
        sims_all = self.cosine_similarity(x, keys)

        if exclude_self_window and sample_ids is not None:
            self_range = torch.arange(-self.seq_len, self.seq_len + 1, device=x.device).unsqueeze(0)
            invalid_index = sample_ids.unsqueeze(1) + self_range
            invalid_index = invalid_index // self.stride
            invalid_index[invalid_index < 0] = 0
            invalid_index[invalid_index >= self.index] = self.index - 1
            row_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).repeat(1, invalid_index.size(1))
            sims_all[row_idx, invalid_index] = -100.0

        sims_topk, indices_topk = torch.topk(sims_all, dim=1, k=k)
        probs_topk = torch.softmax(sims_topk, dim=1).unsqueeze(-1).unsqueeze(-1)
        retrieved_values = values[indices_topk]
        output = torch.sum(probs_topk * retrieved_values, dim=1)
        return output, sims_topk, 0
