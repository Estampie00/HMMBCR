import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn


class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features  # num_classes
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        weight_size = self.weight.size(1)
        sqrt_index = math.sqrt(weight_size)
        stdv = 1. / sqrt_index
        for i in range(self.in_features):  # for i in num_classes
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        w = self.weight
        x = input * w
        x = torch.sum(x, -1)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)