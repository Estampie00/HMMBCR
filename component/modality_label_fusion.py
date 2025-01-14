import torch
import torch.nn as nn
import torch.nn.functional as F


class Modality_Label_Fusion(nn.Module):
    def __init__(self, d_model, args):
        super().__init__()
        self.fusion_type = args.m2l_fusion_type  # [add, angleFus]
        self.num_types = args.num_types
        self.node_type = args.node_type
        if self.fusion_type =='Heter_angleFus':
            self.get_score = nn.ModuleList()
            for i in range(self.num_types):
                self.get_score.append(nn.Linear(d_model, 1, bias=False))
        else:
            self.get_score = nn.Linear(d_model, 1, bias=False)

    def forward(self, angle_1, angle_2, angle_3):  # [B,vocab_size,dim]
        if self.fusion_type == 'add':
            output = (angle_1 + angle_2 + angle_3) / 3
            attn_weight = None
        elif self.fusion_type == 'angleFus':
            angle_all = torch.cat([angle_1.unsqueeze(2), angle_2.unsqueeze(2), angle_3.unsqueeze(2)], 2)  # [B,vocab_size,3,dim]
            angle_score = self.get_score(angle_all)
            angle_score = angle_score.transpose(2, 3)  #
            attn_weight = F.softmax(angle_score, dim=-1)
            output = torch.matmul(attn_weight, angle_all).squeeze(-2)
        elif self.fusion_type == 'Heter_angleFus':
            angle_all = torch.cat([angle_1.unsqueeze(2), angle_2.unsqueeze(2), angle_3.unsqueeze(2)], 2)
            lengths = []
            for i in range(len(self.node_type)):
                lengths.append(len(self.node_type[i]))
            angle_all_group = []
            start_idx = 0
            for length in lengths:
                angle_all_group.append(angle_all[:, start_idx:start_idx + length, :, :])
                start_idx += length
            angle_score_group = []
            for i in range(self.num_types):
                angle_score_group.append(self.get_score[i](angle_all_group[i]))
            angle_score = torch.cat(angle_score_group, dim=1)
            angle_score = angle_score.transpose(2, 3)
            attn_weight = F.softmax(angle_score, dim=-1)
            output = torch.matmul(attn_weight, angle_all).squeeze(-2)
        else:
            raise NotImplementedError
        return output, attn_weight