# === Updated EAFlowPredictionNet using Sobel edge and precomputed edge maps ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
class SobelEdgeExtractor(nn.Module):
    def __init__(self):
        super(SobelEdgeExtractor, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, images):

        gray = images.mean(dim=1, keepdim=True)
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edge_map = magnitude / (magnitude.amax(dim=[2, 3], keepdim=True) + 1e-6)

        return edge_map.expand_as(images)  # match input shape (B, C, H, W)


class EAFlowPredictionNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, W, H, device,
                 output_channels=4, edge_method='augmentation'):
        super(EAFlowPredictionNet, self).__init__()

        from ..model_Griffin.flowpred2_griffin import FlowPredictionNet
        self.edge_method = edge_method
        self.edge_weight = nn.Parameter(torch.tensor(0.3))

        self.backbone = FlowPredictionNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            W=W, H=H, device=device,
            output_channels=output_channels
        )

    def forward(self, I1_I2, I3_I4, edge_I1_I2=None, edge_I3_I4=None):
        if self.edge_method == 'none':
            return self.backbone(I1_I2, I3_I4)

        def apply_edge(img_pair, edge_pair):
            if self.edge_method == 'augmentation':
                I1 = img_pair[:, :3]; I2 = img_pair[:, 3:]
                E1 = edge_pair[:, :3]; E2 = edge_pair[:, 3:]
                I1_proc = (1 - self.edge_weight) * I1 + self.edge_weight * (I1 * E1)
                I2_proc = (1 - self.edge_weight) * I2 + self.edge_weight * (I2 * E2)
                return torch.cat([I1_proc, I2_proc], dim=1)
            elif self.edge_method == 'concatenation':
                return torch.cat([img_pair, edge_pair], dim=1)
            else:
                return img_pair

        I1_I2_processed = apply_edge(I1_I2, edge_I1_I2)
        I3_I4_processed = apply_edge(I3_I4, edge_I3_I4)
        return self.backbone(I1_I2_processed, I3_I4_processed)
