# coding : utf-8
# Author : Yuxiang Zeng

import torch
import torch.nn as nn
from tslearn.metrics import dtw
import torch.nn.functional as F

from torchmetrics.functional.pairwise import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_manhattan_distance,
    pairwise_minkowski_distance,
    pairwise_linear_similarity,
)

class PairwiseLoss(nn.Module):
    def __init__(self, method: str = 'cosine', reduction: str = 'mean', **kwargs):
        super().__init__()
        self.method = method.lower()
        self.reduction = reduction
        self.kwargs = kwargs
        self.func = self._get_func()

    def _get_func(self):
        if self.method == 'cosine':
            return lambda x, y: 1 - pairwise_cosine_similarity(x, y)
        elif self.method == 'euclidean':
            return pairwise_euclidean_distance
        elif self.method == 'manhattan':
            return pairwise_manhattan_distance
        elif self.method == 'minkowski':
            return lambda x, y: pairwise_minkowski_distance(x, y, **self.kwargs)
        elif self.method == 'linear':
            return lambda x, y: -pairwise_linear_similarity(x, y)  # 变为 loss
        elif self.method == 'kl':
            # 这样会惩罚 B 偏离 A 的地方，以 A 为“真理”
            return lambda x, y: F.kl_div(
                torch.log(torch.clamp(y / y.sum(dim=-1, keepdim=True), min=1e-10)),
                torch.clamp(x / x.sum(dim=-1, keepdim=True), min=1e-10),
                reduction='sum'
            )
        elif self.method == 'dtw':
            return dtw
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.func(x, y)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# Example for each method
if __name__ == "__main__":
    bs = 2
    dim = 2

    x = torch.randn(bs, dim)
    y = torch.randn(bs, dim)

    print("\n--- Cosine Similarity ---")
    calc_cos = PairwiseLoss(method='cosine')
    print(calc_cos(x, y))

    print("\n--- Euclidean Distance ---")
    calc_euc = PairwiseLoss(method='euclidean')
    print(calc_euc(x, y))

    print("\n--- Manhattan Distance ---")
    calc_man = PairwiseLoss(method='manhattan')
    print(calc_man(x, y))

    print("\n--- Linear Similarity ---")
    calc_lin = PairwiseLoss(method='linear')
    print(calc_lin(x, y))

    print("\n--- Minkowski Distance (p=3) ---")
    calc_mink = PairwiseLoss(method='minkowski', exponent=3)
    print(calc_mink(x, y))

    print("\n--- Dynamic Time Warping ---")
    calc_dtw = PairwiseLoss(method='dtw')
    print(calc_dtw(x, y))

    print("\n--- Kullback-Leibler Divergence ---")
    calc_kl = PairwiseLoss(method='kl')
    print(calc_kl(x, y))
