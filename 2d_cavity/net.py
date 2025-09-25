import  torch
import torch.nn as nn
from collections import OrderedDict


# 定义双精度的全连接神经网络
class FCNet(torch.nn.Module):
    def __init__(self, num_ins=4,  # 根据合并后的输入特征数调整
                 num_outs=3,
                 num_layers=10,
                 hidden_size=50,
                 activation=torch.nn.Tanh):
        """
        初始化全连接神经网络模型。

        参数:
        - num_ins: 输入特征的数量，默认为4（合并后的 x 和 y）。
        - num_outs: 输出特征的数量，默认为3。
        - num_layers: 隐藏层的数量，默认为10。
        - hidden_size: 隐藏层的神经元数量，默认为50。
        - activation: 激活函数，默认为双曲正切函数。
        """
        super(FCNet, self).__init__()

        layers = [num_ins] + [hidden_size] * num_layers + [num_outs]
        self.depth = len(layers) - 1
        self.activation = activation

        layer_list = []
        for i in range(self.depth - 1):
            # 确保每个线性层都是双精度
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]).double()))
            # 激活函数会自动匹配输入的dtype
            layer_list.append(('activation_%d' % i, self.activation()))

        # 最后一层也要转换为双精度
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]).double()))
        layerDict = OrderedDict(layer_list)

        # 将所有层组合为Sequential，并确保转换为双精度
        self.layers = torch.nn.Sequential(layerDict).double()

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 输入张量，必须是双精度。

        返回:
        - out: 神经网络的输出，双精度。
        """
        out = self.layers(x)
        return out

