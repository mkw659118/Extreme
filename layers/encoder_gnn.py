# coding : utf-8
# Author : Yuxiang Zeng
import torch
import dgl
from dgl.nn.pytorch import *
import numpy as np


class GnnFamily(torch.nn.Module):
    def __init__(self, rank, order, graph_encoder):
        super(GnnFamily, self).__init__()
        self.order = order
        self.rank = rank
        self.graph_encoder = graph_encoder
        self.seq_encoder = torch.nn.Linear(1, self.rank)
        if graph_encoder == 'gcn':
            self.layers = torch.nn.ModuleList([GraphConv(self.rank, self.rank) for i in range(self.order)])
        elif graph_encoder == 'graphsage':
            self.layers = torch.nn.ModuleList([SAGEConv(self.rank, self.rank, aggregator_type='gcn') for i in range(self.order)])
        elif graph_encoder == 'gat':
            self.layers = torch.nn.ModuleList([GATConv(self.rank, self.rank, 2, 0.10) for i in range(self.order)])
        elif graph_encoder == 'gin':
            self.layers = torch.nn.ModuleList([GINConv(GINMLP(self.rank, self.rank), 'sum')])
        else:
            raise NotImplementedError
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(self.rank) for _ in range(self.order)])
        self.acts = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(self.order)])
        self.dropout = torch.nn.Dropout(0.10)
        self.readout_layer = torch.nn.Linear(self.rank, self.rank)
        self.number_of_nodes = 5
        self.pred_layer = torch.nn.Linear(self.rank * self.number_of_nodes, num_classes)


    def forward(self, graph):
        feats = graph.ndata['feats'].reshape(-1, 1)
        bs = len(feats) // self.number_of_nodes
        feats = self.seq_encoder(feats)
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(graph, feats)
            if self.graph_encoder == 'gat':
                feats = feats.mean(dim=1)  # 聚合多个头的输出
            feats = norm(feats)
            feats = act(feats)
            if self.graph_encoder != 'gat':
                feats = self.dropout(feats)
        feats = self.readout_layer(feats)
        feats = feats.reshape(bs, -1)
        y = self.pred_layer(feats)
        return y


class GINMLP(torch.nn.Module):
    def __init__(self,in_feats,out_feats=64, layer_nums = 3):
        super(GINMLP,self).__init__()
        self.linear_layers = torch.nn.ModuleList()
        for each in range(layer_nums):
            if each == 0 :
                in_features= in_feats
            else:
                in_features = out_feats
            self.linear_layers.append(torch.nn.Linear(in_features= in_features,out_features=out_feats))
        self.activate = torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm1d(out_feats)
        self.dropout = torch.nn.Dropout(p=0.0)

    def forward(self, x):
        x1 = x
        for mod in self.linear_layers :
            x1 = mod(x1)
            x1 = self.activate(x1)
        x2 = self.batchnorm(x1)
        x3 = self.dropout(x2)
        return x3

if __name__ == '__main__':

    # 设置超参数
    num_nodes = 5  # 5 个节点
    num_edges = 6  # 假设有 6 条边
    rank = 64  # 特征维度
    max_flow_length = 5  # 假设最大流长度为 5
    num_classes = 5  # 假设分类任务有 2 类

    # 构建节点特征，随机生成节点特征 图
    feats = torch.randn(num_nodes, 1)  # 5 个节点，每个节点有 1 维特征
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [1, 4]])  # 连接节点的边
    src, dst = torch.tensor(edges[:, 0]), torch.tensor(edges[:, 1])
    graph = dgl.graph((src, dst))
    graph.ndata['feats'] = feats  # 将节点特征添加到图中
    # 创建模型实例
    graph_encoder = 'gcn'  # 选择一种图编码器，例如 'gcn'、'graphsage'、'gat' 或 'gin'
    model = GnnFamily(rank=rank, order=3, graph_encoder=graph_encoder)
    output = model(graph)
    print(output)

    graph_encoder = 'gat'  # 选择一种图编码器，例如 'gcn'、'graphsage'、'gat' 或 'gin'
    model = GnnFamily(rank=rank, order=3, graph_encoder=graph_encoder)
    output = model(graph)
    print(output)

    graph_encoder = 'gin'  # 选择一种图编码器，例如 'gcn'、'graphsage'、'gat' 或 'gin'
    model = GnnFamily(rank=rank, order=3, graph_encoder=graph_encoder)
    output = model(graph)
    print(output)