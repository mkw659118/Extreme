import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 定义RNN层
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True)

        # 定义全连接层，将RNN的输出映射到输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        # 前向传播通过RNN
        out, hn = self.rnn(x, h0)

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])

        return out


if __name__ == '__main__':
    # 设置超参数
    input_dim = 128  # 输入特征的维度
    hidden_dim = 256  # 隐藏层维度
    output_dim = 1  # 输出维度（标量）
    n_layers = 2  # RNN的层数
    batch_size = 32  # 批量大小
    seq_len = 10  # 序列长度

    # 创建模型实例
    model = RNN(input_dim, hidden_dim, output_dim, n_layers=n_layers)

    # 构造输入样例
    x = torch.randn(batch_size, seq_len, input_dim)  # 随机生成输入数据

    # 通过模型进行前向传播
    output = model(x)

    # 输出结果的形状
    print("Output shape:", output.shape)