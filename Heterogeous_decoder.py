import copy
from typing import Optional, Any, Union, Callable, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from component.modality_label_fusion import Modality_Label_Fusion


class DecoderLayer(Module):
    """
    We do not use self-attention in our Heterogeneous Decoder
    """
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, num_types: int,
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False) -> None:
        super(DecoderLayer, self).__init__()
        self.num_types = num_types
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.multi_head_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(DecoderLayer, self).__setstate__(state)

    def forward(self, tgt: list, memory: Tensor, memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tuple:

        x = tgt
        attn_list = []
        if self.norm_first:
            for i in range(self.num_types):
                x_h, attn = self._mha_block(self.norm1(x[i]), memory, memory_mask, memory_key_padding_mask)
                x[i] = x[i] + x_h
                attn_list.append(attn)
            x_cat = torch.cat(x, dim=0)
            x_cat = x_cat + self._ff_block(self.norm2(x_cat))
            attn_cat = torch.cat(attn_list, dim=1)
        else:
            for i in range(self.num_types):
                x_h, attn = self._mha_block(x[i], memory, memory_mask, memory_key_padding_mask)
                x[i] = self.norm1(x[i] + x_h)
                attn_list.append(attn)
            x_cat = torch.cat(x, dim=0)
            x_cat = self.norm2(x_cat + self._ff_block(x_cat))
            attn_cat = torch.cat(attn_list, dim=1)
        return x_cat, attn_cat

    # multi-head attention block
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tuple:
        x, attn = self.multi_head_attn(x, mem, mem,
                                       attn_mask=attn_mask,
                                       key_padding_mask=key_padding_mask,
                                       need_weights=True)
        return self.dropout2(x), attn

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class Heterogeneous_LabelDecoder(nn.Module):
    def __init__(self, args, d_model):
        super(Heterogeneous_LabelDecoder, self).__init__()
        self.classes = args.num_classes
        self.num_types = args.num_types
        self.node_type = args.node_type
        # time-series data
        self.layer_motion_stack = nn.ModuleList()
        for _ in range(args.dec_num_layers):
            self.layer_motion_stack.append(DecoderLayer(d_model, args.dec_heads, 4 * d_model, args.num_types,
                                                        args.dec_dropout, args.dec_activation,
                                                        args.dec_layer_norm_eps, args.dec_norm_first))

        self.layer_audio_stack = nn.ModuleList()
        for _ in range(args.dec_num_layers):
            self.layer_audio_stack.append(DecoderLayer(d_model, args.dec_heads, 4 * d_model, args.num_types,
                                                       args.dec_dropout, args.dec_activation,
                                                       args.dec_layer_norm_eps, args.dec_norm_first))

        # discrete_state data (ECA_Net)
        self.layer_state_attn = nn.ModuleList()
        for i in range(self.num_types):
            self.layer_state_attn.append(nn.Conv1d(kernel_size=args.dec_state_kernel, stride=1, in_channels=1,
                                                   out_channels=len(self.node_type[i]), padding=(args.dec_state_kernel - 1) // 2,
                                                   bias=False))

        # calculate the fusion score
        self.multi_angle_fus = Modality_Label_Fusion(d_model, args)

    def forward(self, tgt_input, enc_motion, enc_audio, enc_state):
        # tgt_input is a list
        # per modal dec
        dec_motion, dec_audio, dec_state = None, None, None

        for idx, dec_layer in enumerate(self.layer_motion_stack):
            dec_motion, _ = dec_layer(tgt_input, enc_motion)

        for idx, dec_layer in enumerate(self.layer_audio_stack):
            dec_audio, _ = dec_layer(tgt_input, enc_audio)

        dec_enc_state_attn_heter = []
        dec_state_heter = []

        for i in range(self.num_types):
            dec_enc_state_attn = torch.sigmoid(self.layer_state_attn[i](enc_state.permute(1, 0, 2)))
            dec_enc_state_attn_heter.append(dec_enc_state_attn)
            dec_state_heter.append(enc_state.expand(len(self.node_type[i]), -1, -1).permute(1, 0, 2) * dec_enc_state_attn)

        dec_state = torch.cat(dec_state_heter, dim=1)

        # m2l fusion
        dec_output, _ = self.multi_angle_fus(dec_motion.permute(1, 0, 2), dec_audio.permute(1, 0, 2), dec_state)  # [N,L,D]

        return dec_output