import torch
import torch.nn as nn
from .blocks_griffin import EncodingBlock, DecodingBlock  # Adjust import path
from .GAP_2 import GAPSupport as GAP

class FlowPredictionNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, W, H, device, output_channels=4):
        super(FlowPredictionNet, self).__init__()
        # Encoder with EncodingBlock

        self.down1 = EncodingBlock(in_channels, hidden_channels, kernel_size)
        self.down2 = EncodingBlock(hidden_channels, hidden_channels * 2, kernel_size)
        self.down3 = EncodingBlock(hidden_channels * 2, hidden_channels * 4, kernel_size)

        # Decoder with DecodingBlock
        self.up1 = DecodingBlock(hidden_channels * 4, hidden_channels * 2, kernel_size)
        self.up2 = DecodingBlock(hidden_channels * 2, hidden_channels, kernel_size)
        self.up3 = DecodingBlock(hidden_channels, hidden_channels // 2, kernel_size)

        # Final optical flow prediction
        self.flow_pred = nn.Conv2d(hidden_channels, out_channels=output_channels, kernel_size=3, padding=1) #merge with time dimension to channel dimension so this layer input chanell change from hid//2 to hid only
        #GAP
        # self.gap = GAP(2,2) #gap(2), fc(2)

    def forward(self, I0, I1):
        """
        Args:
            I0, I1: Input frames (batch_size, channels, height, width)
        Returns:
            F_0_1, F_1_0: Optical flow from I0 to I1 and I1 to I0.
        """
        # Combine frames into a sequence
        seq = torch.stack([I0, I1], dim=1)  # Shape: (batch_size, time, channels, height, width)
        # print('shape of time at flowpred2: ', seq.shape[1])

        # Encoder
        s1, res1 = self.down1(seq)
        s2, res2 = self.down2(s1)
        s3, res3 = self.down3(s2)

        # Decoder
        x = self.up1(s3, res3)
        x = self.up2(x, res2)
        x = self.up3(x, res1)

        # Predict optical flow
        b, t, c, h, w = x.shape
        # x = x.reshape(b * t, c, h, w)  # Merge time*batch dimension for final conv
        # x = x.reshape(b, c*t, h, w)  # Merge time*channel dimension for final conv
        x = x.reshape(b, c * t, h, w)  # Merge time*channel dimension for final conv

        x = self.flow_pred(x)  # Channel x = 4 (f10x,f10y,f01x,f01y)

        return x
