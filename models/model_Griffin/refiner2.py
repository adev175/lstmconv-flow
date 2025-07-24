from .flowpred2 import FlowPredictionNet
from .GAP_2 import GAPSupport
from torch import nn

from .flowpred2_griffin import FlowPredictionNet  # Import the FlowPredictionNet class

class NetRefiner(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, W, H, device):
        super(NetRefiner, self).__init__()
        # Initialize the flow prediction network with all required arguments
        self.flow_model = FlowPredictionNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            W=W,
            H=H,
            device=device
        )  # Predict F_0_1 and F_1_0

        self.gap_support = GAPSupport(in_channels=2)  # GAP for 2 flow channels
        # self.refine_model = RefinementNet()  # Refine interpolated frames

    def forward(self, I0, I1):
        """
        Args:
            I0, I1: Input flow

        Returns:
            Interpolated frame It.
        """
        # Predict forward and backward flows
        F_0_1, F_1_0 = self.flow_model(I0, I1)  # Shape: (batch_size*2, 2, height, width)
        # print(f"shape of F_0_1: {F_0_1.shape}")

        # Compute support vector for F_0_1 and F_1_0
        support_0_1 = self.gap_support(F_0_1)  # Shape: (batch_size*2, 2)
        support_1_0 = self.gap_support(F_1_0)  # Shape: (batch_size*2, 2)

        # Expand and add support vector back to the flows
        F_0_1 = F_0_1 + support_0_1.unsqueeze(2).unsqueeze(3)  # Add to flow map
        F_1_0 = F_1_0 + support_1_0.unsqueeze(2).unsqueeze(3)

        return F_0_1, F_1_0
