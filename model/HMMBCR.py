import torch
import torch.nn as nn
import argparse
from utils.opts import get_args
from module.feature_extraction import feature_extraction
from module.MULT import MULTModel
from module.Heterogeous_decoder import Heterogeneous_LabelDecoder
from module.HGAT import HGAT
from component.per_label_classifiers import Element_Wise_Layer

parser = argparse.ArgumentParser()
args = get_args(parser)


class HMMBCR(nn.Module):
    def __init__(self, args):
        super(HMMBCR, self).__init__()
        self.adjacency_matrix = torch.tensor(args.adjacency_matrix)
        self.num_classes = args.num_classes
        self.d_model = args.d_model
        self.num_types = args.num_types
        self.node_type = args.node_type
        self.feature_root = int(self.d_model / 2)  # bool
        self.bias = args.classifier_bias  # bool

        # tokenization
        self.feature_extractor = feature_extraction(self.feature_root, args)

        # multi-modal encoder
        self.fusion_model = MULTModel(args)

        # m2l and l2l
        self.label_embedding = nn.ModuleList()
        for i in range(self.num_types):
            self.label_embedding.append(nn.Embedding(num_embeddings=self.node_type[i].shape[0], embedding_dim=2 * self.d_model))
        self.modal_to_label = Heterogeneous_LabelDecoder(args, d_model=2 * self.d_model)
        self.label_to_label = HGAT(args, graph_dmodel=2 * self.d_model, graph_hid=self.d_model, graph_outdim=2 * self.d_model)

        self.classifiers = Element_Wise_Layer(self.num_classes, 4 * self.d_model, self.bias)

    def forward(self, batch_acc, batch_gyro, batch_audio, batch_state):
        batch_size = batch_acc.shape[0]
        # tokenization
        motion, audio, state = self.feature_extractor(batch_acc, batch_gyro, batch_audio, batch_state)  # [B, D, L]
        src_motion = motion.permute(2, 0, 1)
        src_audio = audio.permute(2, 0, 1)
        src_state = state.unsqueeze(0)  # [L, N, D]

        # fusion
        enc_motion, enc_audio, enc_state = self.fusion_model(src_motion, src_audio, src_state)  # [L,N,D]

        # modality-label dependence
        # categorization
        tgt_input = []
        for i in range(self.num_types):
            tgt_input.append(self.label_embedding[i].weight.unsqueeze(1).expand(-1, batch_size, -1))  # [L,N,D]
        m2l_output, _ = self.modal_to_label(tgt_input, enc_motion, enc_audio, enc_state)

        # label-label dependence
        # categorization
        l2l_output, _, _, _ = self.label_to_label(m2l_output, self.adjacency_matrix.to(self.device))  # expected [N, nodes, dim]
        output = torch.cat((m2l_output.view(batch_size * self.num_classes, -1), l2l_output.view(batch_size * self.num_classes, -1)), 1)
        output = output.view(batch_size, self.num_classes, -1)  # [N, L, D]

        result = torch.sigmoid(self.classifiers(output))
        return result
