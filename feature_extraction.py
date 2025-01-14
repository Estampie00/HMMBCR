import torch.nn as nn


class feature_extraction(nn.Module):
    def __init__(self, feature_root, args):
        super(feature_extraction, self).__init__()
        if args.feature_extractor_activation == 'relu':
            self.activation = nn.ReLU()
        elif args.feature_extractor_activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(0.2)

        self.accelerator = nn.Sequential(
            nn.Conv1d(kernel_size=args.motion_ks_1, stride=args.motion_stride, in_channels=3, out_channels=feature_root),
            nn.BatchNorm1d(feature_root),
            self.activation,
            self.dropout,
            nn.Conv1d(kernel_size=args.motion_ks_2, stride=args.motion_stride, in_channels=feature_root, out_channels=2 * feature_root),
            nn.BatchNorm1d(2 * feature_root),
            self.activation,
            self.dropout
        )

        self.gyro = nn.Sequential(
            nn.Conv1d(kernel_size=args.motion_ks_1, stride=args.motion_stride, in_channels=3, out_channels=feature_root),
            nn.BatchNorm1d(feature_root),
            self.activation,
            nn.Conv1d(kernel_size=args.motion_ks_2, stride=args.motion_stride, in_channels=feature_root, out_channels=2 * feature_root),
            nn.BatchNorm1d(2 * feature_root),
            self.activation,
            self.dropout
        )

        self.audio = nn.Sequential(
            nn.Conv1d(kernel_size=args.audio_ks_1, stride=args.audio_stride, in_channels=13, out_channels=feature_root),
            nn.BatchNorm1d(feature_root),
            self.activation,
            self.dropout,
            nn.Conv1d(kernel_size=args.audio_ks_2, stride=args.audio_stride, in_channels=feature_root, out_channels=2 * feature_root),
            nn.BatchNorm1d(2 * feature_root),
            self.activation,
            self.dropout
        )

        self.state = nn.Sequential(
            nn.Linear(in_features=26, out_features=2 * feature_root, bias=False),
            self.activation,
            self.dropout
        )

    def forward(self, acc, gyro, audio, state):
        x_acc = self.accelerator(acc)
        x_gyro = self.gyro(gyro)
        x_audio = self.audio(audio)
        x_state = self.state(state)
        x_motion = x_acc + x_gyro
        return x_motion, x_audio, x_state
