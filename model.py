import torch
import torch.nn as nn

# ==============================
# CNN Encoder
# ==============================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # H/8, W/4; preserve width
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (seq_len=W', batch=B, feature_dim=C'*H')
        """
        out = self.conv_layers(x)
        b, c, h, w = out.size()

        # Permute to (W, B, H*C)
        out = out.permute(3, 0, 2, 1).contiguous()
        out = out.reshape(w, b, c * h)  # shape: (seq_len, batch, feature_dim)
        return out

# ==============================
# CRNN Model (CNN + RNN)
# ==============================
class CRNN(nn.Module):
    def __init__(self, num_classes, in_channels=1, rnn_hidden_size=256, rnn_layers=2):
        super(CRNN, self).__init__()
        self.encoder = CNNEncoder(in_channels=in_channels)
        self.rnn_hidden_size = rnn_hidden_size

        # RNN expects input: (seq_len, batch, feature_dim)
        self.rnn = nn.LSTM(
            input_size=256*4,   # CNN output feature_dim (256 channels * height 4)
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True
        )

        self.fc = nn.Linear(rnn_hidden_size*2, num_classes)  # bidirectional → 2*rnn_hidden_size

    def forward(self, x):
        # CNN feature extraction
        conv_out = self.encoder(x)  # (seq_len, batch, feature_dim)

        # RNN sequence modeling
        recurrent_out, _ = self.rnn(conv_out)  # (seq_len, batch, hidden*2)

        # Linear classifier for each time step
        output = self.fc(recurrent_out)  # (seq_len, batch, num_classes)
        return output
