# coding : utf-8
# Author : Yuxiang Zeng
import torch



class Predictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=3, init_method='xavier'):
        super(Predictor, self).__init__()
        self.init_method = init_method
        layers = []
        current_dim = hidden_dim

        # Add the first layer
        layers.append(torch.nn.Linear(input_dim, current_dim))
        layers.append(torch.nn.LayerNorm(current_dim))
        layers.append(torch.nn.LeakyReLU(negative_slope=1e-2))

        # Add intermediate layers
        for _ in range(n_layer - 1):
            next_dim = max(current_dim // 2, output_dim)  # Ensure it doesn't drop below output_dim
            layers.append(torch.nn.Linear(current_dim, next_dim))
            layers.append(torch.nn.LayerNorm(next_dim))
            layers.append(torch.nn.LeakyReLU(negative_slope=1e-2))
            current_dim = next_dim

        # Add the final layer
        layers.append(torch.nn.Linear(current_dim, output_dim))

        # Combine layers into a sequential module
        self.mlp = torch.nn.Sequential(*layers)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of the layers based on the specified method.
        Available methods: 'xavier', 'kaiming', 'normal'.
        """
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                if self.init_method == 'xavier':
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.constant_(module.bias, 0)
                elif self.init_method == 'kaiming':
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    torch.nn.init.constant_(module.bias, 0)
                elif self.init_method == 'normal':
                    torch.nn.init.normal_(module.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(module.bias, 0)
                else:
                    raise ValueError(f"Unknown initialization method: {self.init_method}")

    def forward(self, x):
        return self.mlp(x)

if __name__ == '__main__':
    # 设置超参数
    input_dim = 128  # 输入特征的维度
    hidden_dim = 256  # 隐藏层维度
    output_dim = 1  # 输出维度（标量）
    n_layer = 3  # 网络的层数
    batch_size = 32  # 批量大小

    # 创建模型实例
    model = Predictor(input_dim, hidden_dim, output_dim, n_layer=n_layer, init_method='xavier')

    # 构造输入样例
    x = torch.randn(batch_size, input_dim)  # 随机生成输入数据

    # 通过模型进行前向传播
    output = model(x)

    # 输出结果的形状
    print("Output shape:", output.shape)
