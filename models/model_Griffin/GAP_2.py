import torch.nn as nn
import torch
from .blocks_griffin import ChannelAttention, SpatialAttention, AttentionBlock
class GlobalAttentionSystem(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2, out_channels=1):
        super(GlobalAttentionSystem, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio) #in channel must > reduction ratio
        self.attention_block = AttentionBlock(in_channels)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.channel_attention(x)
        x = self.attention_block(x)
        x = self.final_conv(x)
        return x

class GAPSupport(nn.Module): #main called function
    def __init__(self, in_channels, reduction_ratio=2):
        super(GAPSupport, self).__init__()
        self.global_pool = GlobalAttentionSystem(in_channels, reduction_ratio, in_channels)  # GAP
        # self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)  # Convolutional layer to refine the descriptor
        self.fc = nn.Linear(in_channels, in_channels)  # Fully connected layer to refine the descriptor #2 channel: x,y flow

    def forward(self, flow_map):
        """
        Args:
            flow_map: Optical flow map (batch_size, channels, height, width).

        Returns:
            support_vector: A global descriptor (batch_size, channels).
        """
        b, c, h, w = flow_map.size()
        pooled = self.global_pool(flow_map).view(b, c)  # GAP over height and width
        support_vector = self.fc(pooled)  # Refine support vector
        return support_vector

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        F_in = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        A_s = torch.sigmoid(x)
        return F_in * A_s