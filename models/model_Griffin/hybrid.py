import torch.nn as nn
from .coffin import GatedConvLSTM, LocalAttention
class HybridBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3):
        super(HybridBlock, self).__init__()
        self.gated_conv_lstm = GatedConvLSTM(in_chans, out_chans, kernel_size)
        self.local_attention = LocalAttention(out_chans, kernel_size)

    def forward(self, x, prev_state):
        x, new_state = self.gated_conv_lstm(x, prev_state)
        x = self.local_attention(x)
        return x, new_state