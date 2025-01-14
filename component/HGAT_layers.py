import torch
import torch.nn as nn
import torch.nn.functional as F


class HGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, node_type, graph_regular_lammba):
        super(HGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.node_type = node_type
        self.graph_regular_lammba = graph_regular_lammba

        self.att_node_type = nn.ModuleList()  # l2c
        self.att_node_node = nn.ModuleList()  # l2l
        for i in range(len(self.node_type)):
            for j in range(len(self.node_type)):
                self.att_node_type.append(Attention_node_type(out_features, gamma=self.alpha))
            self.att_node_node.append(Attention_node_node(out_features, gamma=self.alpha))

        self.W_list = nn.ParameterList()
        for num in range(len(self.node_type)):
            W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)), requires_grad=True)
            nn.init.xavier_uniform_(W.data, gain=1.414)
            self.W_list.append(W)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        :param h: [N, N_nodes, dim]
        :param adj: [N, N_nodes, N_nodes]
        :return:
        """
        adj = adj.float()
        lengths = []
        for i in range(len(self.node_type)):
            lengths.append(len(self.node_type[i]))
        h_list = []
        adj_group_list = []
        start_idx = 0
        for length in lengths:
            h_list.append(h[:, start_idx:start_idx + length, :])
            adj_group_list.append(adj[:, start_idx:start_idx + length])
            start_idx += length

        Wh_list, node_embedding_list = [], []
        for j in range(len(self.node_type)):
            Wh = torch.matmul(h_list[j], self.W_list[j])
            h_t_all = torch.matmul(adj_group_list[j], Wh)
            h_t_all_list = []
            start_IDX = 0
            for length in lengths:
                h_t_all_list.append(h_t_all[:, start_IDX:start_IDX + length, :])
                start_IDX += length
            node_embedding_list.append(h_t_all_list)
            Wh_list.append(Wh)

        node_type_att_list = []
        num = 0
        for m in range(len(self.node_type)):
            each_node_type_att_list = []
            for n in range(len(self.node_type)):
                each_node_each_type_att = self.att_node_type[num](Wh_list[m], node_embedding_list[n][m])
                each_node_type_att_list.append(each_node_each_type_att)
                num += 1
            each_node_type_att_array = F.softmax(torch.cat(each_node_type_att_list, dim=2).squeeze(-1), dim=-1)  # normalization
            node_type_att_list.append(each_node_type_att_array)

        node_type_att_matrix = torch.cat(node_type_att_list, dim=1)

        original_node_node_att_list = []
        node_node_att_list = []
        for o in range(len(self.node_type)):
            per_type_original_att_list = []
            per_type_node_node_att_list = []
            for p in range(len(self.node_type)):
                node_node_att = self.att_node_node[o](Wh_list[o], Wh_list[p])
                node_type_att = node_type_att_list[o][:, :, p].unsqueeze(-1).repeat(1, 1, node_node_att.size()[-1])
                node_node_type_att = node_node_att * node_type_att  #
                per_type_node_node_att_list.append(node_node_type_att)
                per_type_original_att_list.append(node_node_att)
            node_node_att_list.append(torch.cat(per_type_node_node_att_list, dim=2))
            original_node_node_att_list.append(torch.cat(per_type_original_att_list, dim=2))

        # original
        original_node_node_att_matrix = torch.cat(original_node_node_att_list, dim=1)
        original_zero_vec = -9e15 * torch.ones_like(original_node_node_att_matrix)
        original_attention = torch.where(adj > 0, original_node_node_att_matrix, original_zero_vec)
        original_attention = F.softmax(original_attention, dim=-1)

        # heter
        node_node_att_matrix = torch.cat(node_node_att_list, dim=1)
        zero_vec = -9e15 * torch.ones_like(node_node_att_matrix)
        attention = torch.where(adj > 0, node_node_att_matrix, zero_vec)
        attention = F.softmax(attention, dim=-1)

        regular_att_matrix = ((1 - self.graph_regular_lammba) * adj + self.graph_regular_lammba * attention).float()
        type_n = [len(label_type) for label_type in self.node_type]
        attention_matrix = [regular_att_matrix[:, :att_type, :] for att_type in type_n]

        h_output_list = []
        Wh_matrix = torch.cat(Wh_list, dim=1)
        for s in range(len(self.node_type)):
            forward_attention = attention_matrix[s]
            h_node = torch.matmul(forward_attention, Wh_matrix)
            h_output_list.append(h_node)
        h_output = torch.cat(h_output_list, dim=1)

        return h_output, node_type_att_matrix, attention, original_attention


class Attention_node_node(nn.Module):
    def __init__(self, dim_features, gamma=0.1):
        super(Attention_node_node, self).__init__()
        self.dim_features = dim_features
        self.a = nn.Parameter(torch.empty(size=(2 * dim_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.gamma = gamma
        self.leakyrelu = nn.LeakyReLU(self.gamma)

    def forward(self, input1, input2):
        Wh1 = torch.matmul(input1, self.a[:self.dim_features, :])
        Wh2 = torch.matmul(input2, self.a[self.dim_features:, :])
        # broadcast add
        # e = Wh1 + Wh2.T
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)


class Attention_node_type(nn.Module):
    def __init__(self, dim_features, gamma=0.1):
        super(Attention_node_type, self).__init__()
        self.dim_features = dim_features
        self.a = nn.Parameter(torch.empty(size=(2 * dim_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.gamma = gamma
        self.leakyrelu = nn.LeakyReLU(self.gamma)

    def forward(self, input1, input2):
        Wh_calculate = torch.cat([input1, input2], dim=2)  # bs, num1, 2*dim
        e = torch.matmul(Wh_calculate, self.a)
        # broadcast add
        # e = Wh1 + Wh2.T
        return self.leakyrelu(e)