# 1. Edge Extraction using Canny (exact from paper)
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# 2. Modified FlowPredictionNet with Edge-Aware (minimal change)

class FlowPredictionNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, W, H, device,
                 output_channels=4, edge_method='augmentation'):
        super(FlowPredictionNet, self).__init__()

        # Keep your existing architecture
        from .blocks_griffin import EncodingBlock, DecodingBlock

        self.edge_method = edge_method
        self.edge_extractor = CannyEdgeExtractor()

        # Adjust input channels for concatenation method
        if edge_method == 'concatenation':
            effective_in_channels = in_channels * 2  # RGB + Edge RGB = 6 channels
        else:
            effective_in_channels = in_channels

        # Your existing encoder-decoder
        self.down1 = EncodingBlock(effective_in_channels, hidden_channels, kernel_size)
        self.down2 = EncodingBlock(hidden_channels, hidden_channels * 2, kernel_size)
        self.down3 = EncodingBlock(hidden_channels * 2, hidden_channels * 4, kernel_size)

        self.up1 = DecodingBlock(hidden_channels * 4, hidden_channels * 2, kernel_size)
        self.up2 = DecodingBlock(hidden_channels * 2, hidden_channels, kernel_size)
        self.up3 = DecodingBlock(hidden_channels, hidden_channels // 2, kernel_size)

        self.flow_pred = nn.Conv2d(hidden_channels, output_channels, 3, padding=1)

        # Two-stream: duplicate network for edge processing
        if edge_method == 'two_stream':
            self.edge_down1 = EncodingBlock(in_channels, hidden_channels, kernel_size)
            self.edge_down2 = EncodingBlock(hidden_channels, hidden_channels * 2, kernel_size)
            self.edge_down3 = EncodingBlock(hidden_channels * 2, hidden_channels * 4, kernel_size)

            self.edge_up1 = DecodingBlock(hidden_channels * 4, hidden_channels * 2, kernel_size)
            self.edge_up2 = DecodingBlock(hidden_channels * 2, hidden_channels, kernel_size)
            self.edge_up3 = DecodingBlock(hidden_channels, hidden_channels // 2, kernel_size)

            self.edge_flow_pred = nn.Conv2d(hidden_channels, output_channels, 3, padding=1)

    def forward(self, I0, I1):
        """Apply edge-aware mechanisms exactly as in paper"""

        if self.edge_method == 'augmentation':
            # Edge Augmentation: I_aug = 1/2 * (I + I ⊙ E)
            E0 = self.edge_extractor(I0)
            E1 = self.edge_extractor(I1)

            I0_processed = 0.5 * (I0 + I0 * E0)
            I1_processed = 0.5 * (I1 + I1 * E1)

            seq = torch.stack([I0_processed, I1_processed], dim=1)

            # Forward through main network
            s1, res1 = self.down1(seq)
            s2, res2 = self.down2(s1)
            s3, res3 = self.down3(s2)

            x = self.up1(s3, res3)
            x = self.up2(x, res2)
            x = self.up3(x, res1)

            b, t, c, h, w = x.shape
            x = x.reshape(b, c * t, h, w)
            flow = self.flow_pred(x)

        elif self.edge_method == 'concatenation':
            # Edge Concatenation: I_con = [I; I ⊙ E]
            E0 = self.edge_extractor(I0)
            E1 = self.edge_extractor(I1)

            I0_processed = torch.cat([I0, I0 * E0], dim=1)  # 6 channels
            I1_processed = torch.cat([I1, I1 * E1], dim=1)  # 6 channels

            seq = torch.stack([I0_processed, I1_processed], dim=1)

            # Forward through main network
            s1, res1 = self.down1(seq)
            s2, res2 = self.down2(s1)
            s3, res3 = self.down3(s2)

            x = self.up1(s3, res3)
            x = self.up2(x, res2)
            x = self.up3(x, res1)

            b, t, c, h, w = x.shape
            x = x.reshape(b, c * t, h, w)
            flow = self.flow_pred(x)

        elif self.edge_method == 'two_stream':
            # Two-Stream: F = 1/2 * (F_I + F_E)

            # Frame stream
            seq_frame = torch.stack([I0, I1], dim=1)
            s1, res1 = self.down1(seq_frame)
            s2, res2 = self.down2(s1)
            s3, res3 = self.down3(s2)
            x = self.up1(s3, res3)
            x = self.up2(x, res2)
            x = self.up3(x, res1)
            b, t, c, h, w = x.shape
            x = x.reshape(b, c * t, h, w)
            F_I = self.flow_pred(x)

            # Edge stream
            E0 = self.edge_extractor(I0)
            E1 = self.edge_extractor(I1)
            seq_edge = torch.stack([E0, E1], dim=1)

            s1, res1 = self.edge_down1(seq_edge)
            s2, res2 = self.edge_down2(s1)
            s3, res3 = self.edge_down3(s2)
            x = self.edge_up1(s3, res3)
            x = self.edge_up2(x, res2)
            x = self.edge_up3(x, res1)
            b, t, c, h, w = x.shape
            x = x.reshape(b, c * t, h, w)
            F_E = self.edge_flow_pred(x)

            # Combine: F = 1/2 * (F_I + F_E)
            flow = 0.5 * (F_I + F_E)

        return flow


class CannyEdgeExtractor(nn.Module):
    """Exact Canny edge extraction as in EA-Net paper"""

    def __init__(self, low_threshold=50, high_threshold=150):
        super(CannyEdgeExtractor, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, images):
        batch_size, channels, height, width = images.shape
        edge_maps = torch.zeros_like(images)

        for b in range(batch_size):
            for c in range(channels):
                img_np = images[b, c].detach().cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)

                # Canny edge detection
                edges = cv2.Canny(img_np, self.low_threshold, self.high_threshold)
                edges = edges.astype(np.float32) / 255.0

                edge_maps[b, c] = torch.from_numpy(edges).to(images.device)

        return edge_maps


# 3. EA-Net Loss Function (exact from paper)
class EANetLoss(nn.Module):
    """Exact loss function from EA-Net paper"""

    def __init__(self):
        super(EANetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.edge_extractor = CannyEdgeExtractor()

    def synthesis_loss(self, pred_frame, gt_frame):
        """L_syn = ||I_t - I_t^g||_1"""
        return self.l1_loss(pred_frame, gt_frame)

    def flow_loss(self, I0, I1, F_0_1, F_1_0):
        """L_flow = ||I0 - warp(I1, F_0_1)||_1 + ||I1 - warp(I0, F_1_0)||_1"""
        warped_I1 = self.warp_image(I1, F_0_1)
        warped_I0 = self.warp_image(I0, F_1_0)

        return self.l1_loss(I0, warped_I1) + self.l1_loss(I1, warped_I0)

    def warp_image(self, img, flow):
        """Bilinear warping function"""
        B, C, H, W = img.size()

        # Create coordinate grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(img.device)

        # Apply flow
        vgrid = grid + flow

        # Normalize to [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        warped = F.grid_sample(img, vgrid, align_corners=True)

        return warped

    def forward(self, pred_frame, gt_frame, I0, I1, F_0_1, F_1_0):
        """Total loss: L = L_syn + L_flow"""
        l_syn = self.synthesis_loss(pred_frame, gt_frame)
        l_flow = self.flow_loss(I0, I1, F_0_1, F_1_0)

        total_loss = l_syn + l_flow

        return total_loss, {
            'synthesis': l_syn.item(),
            'flow': l_flow.item(),
            'total': total_loss.item()
        }


# 4. Simple Integration - just modify your refiner2.py
class NetRefiner(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, W, H, device,
                 edge_method='augmentation'):
        super(NetRefiner, self).__init__()

        # Replace your FlowPredictionNet with edge-aware version
        self.flow_model = FlowPredictionNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            W=W, H=H, device=device,
            edge_method=edge_method  # 'augmentation', 'concatenation', or 'two_stream'
        )

        # Keep your existing GAP
        from .GAP_2 import GAPSupport
        self.gap_support = GAPSupport(in_channels=2)

    def forward(self, I0, I1):
        # Get edge-aware flow
        flow = self.flow_model(I0, I1)  # Shape: (B, 4, H, W)

        # Split into F_0_1 and F_1_0
        F_0_1 = flow[:, :2, :, :]
        F_1_0 = flow[:, 2:, :, :]

        # Apply GAP (keep existing)
        support_0_1 = self.gap_support(F_0_1)
        support_1_0 = self.gap_support(F_1_0)

        F_0_1 = F_0_1 + support_0_1.unsqueeze(2).unsqueeze(3)
        F_1_0 = F_1_0 + support_1_0.unsqueeze(2).unsqueeze(3)

        return F_0_1, F_1_0


# 5. Training usage - minimal change
def train_with_ea_net():
    # Create model with edge method
    model = NetRefiner(
        in_channels=3, hidden_channels=64, kernel_size=3,
        W=256, H=256, device='cuda',
        edge_method='augmentation'  # Start with simplest
    )

    # Use EA-Net loss
    criterion = EANetLoss()

    # In training loop:
    for inputs, targets, _ in dataloader:
        I0, I1 = inputs[0].cuda(), inputs[1].cuda()
        gt_frame = targets[0].cuda()

        # Forward
        F_0_1, F_1_0 = model(I0, I1)

        # Synthesize frame (you need synthesis module)
        pred_frame = synthesis_module(I0, I1, F_0_1, F_1_0)

        # EA-Net loss
        loss, loss_dict = criterion(pred_frame, gt_frame, I0, I1, F_0_1, F_1_0)

        loss.backward()
        optimizer.step()


# 6. Quick test - change only these lines in your code:
"""
1. Replace in refiner2.py:
   - Add edge_method parameter to NetRefiner.__init__()
   - Replace FlowPredictionNet import

2. Replace in training script:
   - criterion = EANetLoss()
   - Use edge-aware loss calculation

That's it! Start with edge_method='augmentation'
"""