import torch.nn as nn
import torch
class GatedConvLSTM(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size):
        super(GatedConvLSTM, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Gates initialization
        self.forget_gate = nn.Conv2d(in_chans + out_chans, out_chans, kernel_size, padding=self.padding)
        self.input_gate = nn.Conv2d(in_chans + out_chans, out_chans, kernel_size, padding=self.padding)
        self.output_gate = nn.Conv2d(in_chans + out_chans, out_chans, kernel_size, padding=self.padding)
        self.cell_gate = nn.Conv2d(in_chans + out_chans, out_chans, kernel_size, padding=self.padding)

    def forward(self, x, prev_state):
        if prev_state is None:
            # Initialize prev_state if it is None
            height, width = x.shape[2], x.shape[3]
            h_prev = torch.zeros((x.shape[0], self.out_chans, height, width), device=x.device)
            c_prev = torch.zeros_like(h_prev)
        else:
            h_prev, c_prev = prev_state

        combined = torch.cat([x, h_prev], dim=1)  # Concatenate input and previous hidden state

        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        g = torch.tanh(self.cell_gate(combined))

        c_curr = f * c_prev + i * g
        h_curr = o * torch.tanh(c_curr)

        return h_curr, (h_curr, c_curr)

class LocalAttention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(LocalAttention, self).__init__()
        self.attention = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        attention_weights = torch.sigmoid(self.attention(x))
        return x * attention_weights

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        ####################
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        ####################

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out