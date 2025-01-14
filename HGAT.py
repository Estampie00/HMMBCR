import torch
import torch.nn as nn
import torch.nn.functional as F
from component.HGAT_layers import HGATLayer


class HGAT(nn.Module):
    def __init__(self, args, graph_dmodel, graph_hid, graph_outdim):
        super(HGAT, self).__init__()
        self.dropout = args.graph_dropout

        self.attentions = [HGATLayer(graph_dmodel, graph_hid, dropout=args.graph_dropout, alpha=args.leaky_alpha,
                                     node_type=args.node_type, graph_regular_lammba=args.graph_regular_lammba) for _ in range(args.gat_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = HGATLayer(graph_hid * args.gat_heads, graph_outdim, dropout=args.graph_dropout, alpha=args.leaky_alpha,
                                 node_type=args.node_type, graph_regular_lammba=args.graph_regular_lammba)

    def forward(self, x, adj):
        """
        :param x: [N, N_nodes, dim]
        :param adj: [N, N_nodes, N_nodes]
        :return:
        """
        x = F.dropout(x, self.dropout, training=self.training)
        x_list = []
        for att in self.attentions:
            x_hid, _, _, _ = att(x, adj)
            x_list.append(x_hid)

        x = torch.cat(x_list, dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x_output, _, _, _ = self.out_att(x, adj)
        x = F.elu(x_output)
        x_list.append(x_output)

        return x





