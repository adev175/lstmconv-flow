import torch.nn.functional as F
import torch.nn as nn
import torch
from .coffin import GatedConvLSTM
from .hybrid import HybridBlock


class EncodingBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3):
        super(EncodingBlock, self).__init__()
        self.conv_lstm = HybridBlock(in_chans, in_chans, kernel_size)
        self.batch_norm_1 = nn.BatchNorm2d(in_chans)
        self.conv2d = nn.Conv2d(in_chans, out_chans, kernel_size, padding=kernel_size // 2)
        self.batch_norm_2 = nn.BatchNorm2d(out_chans)
        self.max_pool = nn.MaxPool2d(2)
        self.out_chans = out_chans

    def forward(self, x):
        b, t, c, h, w = x.shape
        prev_state = None  # Initialize the state with None

        # Process the first image to initialize the state
        x_0, new_state = self.conv_lstm(x[:, 0, :, :, :], prev_state)
        x[:, 0, :, :, :] = x_0

        # Process subsequent images using the updated state from the previous time step
        for time_step in range(1, t):
            x_t, new_state = self.conv_lstm(x[:, time_step, :, :, :], new_state)
            x[:, time_step, :, :, :] = x_t

        # After processing all time steps, reshape and apply further layers
        x = x.reshape(b * t, c, h, w)
        x = F.relu(self.batch_norm_1(x))
        x = self.conv2d(x)
        res = x
        x = self.max_pool(x)
        x = x.reshape(b, t, self.out_chans, h // 2, w // 2)

        return x, res


class DecodingBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3):
        super(DecodingBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)
        self.conv2d_1 = nn.Conv2d(
            in_chans, in_chans, kernel_size, padding=kernel_size // 2
        )
        self.batch_norm_1 = nn.BatchNorm2d(in_chans) #Weight and bias are parameters in the BN layer (they are updated during the back propagation). Running mean and variance are calculated during the forward pass, that's why, I think, they are not considered as parameters (since they do not require gradient).
        self.conv2d_2 = nn.Conv2d(
            in_chans, out_chans, kernel_size, padding=kernel_size // 2
        )
        self.batch_norm_2 = nn.BatchNorm2d(out_chans)

        self.out_chans = out_chans

    def forward(self, x, res):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)

        x = self.up_sample(x)
        x = self.conv2d_1(x) + res
        x = F.relu(self.batch_norm_1(x))
        x = self.conv2d_2(x)
        x = F.relu(self.batch_norm_2(x))

        # reshaping to preserve time-series data
        _, c, h, w = x.shape
        x = x.reshape(b, t, self.out_chans, h, w)

        return x



class ChannelAttention(nn.Module): #for synthesis
    def __init__(self, in_chans, r=3):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        print('in_chans: ', in_chans)
        print('r: ', r)
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, in_chans // r, 1),
            nn.ReLU(),
            nn.Conv2d(in_chans // r, in_chans, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f_in = x
        x = self.gap(x)
        a_c = self.conv(x)

        return f_in * a_c


class SpatialAttention(nn.Module): #for synthesis
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        F_in = x
        avg_out = torch.mean(x, dim=1, keepdim=True)  # channel wise average pool
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # channel wise max pool
        x = torch.cat([avg_out, max_out], dim=1)  # channel-wise concatenation
        x = self.conv(x)
        # A_s = F.sigmoid(x)
        A_s = torch.sigmoid(x)

        return F_in * A_s


class AttentionBlock(nn.Module): #for synthesis
    def __init__(self, num_chans=9, kernel_size=3, padding=1):
        super(AttentionBlock, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_chans, num_chans, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(num_chans, num_chans, kernel_size=kernel_size, padding=padding),
        )
        self.ca_block = ChannelAttention(num_chans,1)
        self.sa_block = SpatialAttention()

    def forward(self, x):
        F_in = x
        x = self.conv_layers(x)
        x = self.ca_block(x)
        x = self.sa_block(x)
        return x + F_in