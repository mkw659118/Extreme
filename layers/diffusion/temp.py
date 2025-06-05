import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from math import pi
from tqdm import tqdm

from layers.diffusion.gaussian_diffusion import *


# 用你原来的 Diffusion 模型代码 & ModelMeanType 引入
# from diffusion_module import GaussianDiffusion, ModelMeanType  ← 你原来的代码

# ========== 模型 ==========
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
        )

    def forward(self, x, t):
        return self.net(x)


# ========== 主函数 ==========
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 构造正弦函数数据：x = linspace(0, 2pi), y = sin(x)
    num_points = 100
    x = np.linspace(0, 2 * pi, num_points)
    y = np.sin(x)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(0).repeat(512, 1)  # [512, 100]

    dataset = TensorDataset(y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 初始化模型和diffusion
    model = MyModel().to(device)
    diffusion = GaussianDiffusion(
        mean_type=ModelMeanType.EPSILON,
        noise_schedule="linear",
        noise_scale=1.0,
        noise_min=1e-4,
        noise_max=0.02,
        steps=1000,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ========== 训练 ==========
    for epoch in range(1000):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x_start = batch[0].to(device)  # [B, 100]
            loss_dict = diffusion.training_losses(model, x_start, reweight=True)
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        # print(f"Epoch {epoch+1} avg loss: {total_loss / len(train_loader):.6f}")

    # ========== 采样 ==========
    with torch.no_grad():
        # z = 0.1 * torch.randn(1, 100).to(device)
        z = torch.randn(1, 100).to(device)
        sampled = diffusion.p_sample(model, z, steps=1000, sampling_noise=True)
        sampled = sampled.squeeze().cpu().numpy()

    # ========== 可视化并保存到本地 ==========
    import matplotlib.pyplot as plt
    import os

    save_dir = "./figures"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(x, np.sin(x), label='Ground Truth')
    plt.plot(x, sampled, label='Sampled (Generated)', linestyle='--')
    plt.legend()
    plt.title("Sine Function Recovery with Diffusion Model")
    plt.savefig(os.path.join(save_dir, "sine_diffusion_result.png"), dpi=300)
    plt.close()
    print("Saved to ./figures/sine_diffusion_result.png")