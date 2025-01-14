import torch
from torch import nn
from component.mult_encoder import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.d_m, self.d_a, self.d_s = args.d_model, args.d_model, args.d_model
        self.num_heads = args.mult_heads
        self.layers = args.mult_num_layers
        self.attn_dropout = args.mult_attn_dropout
        self.relu_dropout = args.mult_relu_dropout
        self.res_dropout = args.mult_res_dropout
        self.out_dropout = args.mult_out_dropout
        self.embed_dropout = args.mult_embed_dropout

        combined_dim = 2 * (self.d_m + self.d_a + self.d_s)

        # 2. Crossmodal Attentions
        self.trans_m_with_a = self.get_network(self_type='ma')
        self.trans_m_with_s = self.get_network(self_type='ms')
        self.trans_a_with_m = self.get_network(self_type='am')
        self.trans_a_with_s = self.get_network(self_type='as')
        self.trans_s_with_m = self.get_network(self_type='sm')
        self.trans_s_with_a = self.get_network(self_type='sa')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_m_mem = self.get_network(self_type='m_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=self.layers)
        self.trans_s_mem = self.get_network(self_type='s_mem', layers=self.layers)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['m', 'am', 'sm']:
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        elif self_type in ['a', 'ma', 'sa']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type in ['s', 'ms', 'as']:
            embed_dim, attn_dropout = self.d_s, self.attn_dropout
        elif self_type == 'm_mem':
            embed_dim, attn_dropout = 2*self.d_m, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 's_mem':
            embed_dim, attn_dropout = 2*self.d_s, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=False)

    def forward(self, x_motion, x_audio, x_state):
        """
        motion, audio, and state should have dimension [seq_len, batch_size, n_features]
        """

        # (audio,state) --> motion
        h_m_with_as = self.trans_m_with_a(x_motion, x_audio, x_audio)    # Dimension (L, N, d_l)
        h_m_with_ms = self.trans_m_with_s(x_motion, x_state, x_state)
        h_ms = torch.cat([h_m_with_as, h_m_with_ms], dim=2)
        h_ms = self.trans_m_mem(h_ms)

        # (motion,state) --> audio
        h_a_with_ms = self.trans_a_with_m(x_audio, x_motion, x_motion)
        h_a_with_as = self.trans_a_with_s(x_audio, x_state, x_state)
        h_as = torch.cat([h_a_with_ms, h_a_with_as], dim=2)
        h_as = self.trans_a_mem(h_as)

        # (audio,motion) --> state
        h_s_with_as = self.trans_s_with_a(x_state, x_audio, x_audio)
        h_s_with_ms = self.trans_s_with_m(x_state, x_motion, x_motion)
        h_ss = torch.cat([h_s_with_as, h_s_with_ms], dim=2)
        h_ss = self.trans_s_mem(h_ss)

        return h_ms, h_as, h_ss






