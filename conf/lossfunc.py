# conf/lossfunc.py - Updated with new loss weighting
import torch
import torch.nn as nn


class EANetLoss(nn.Module):
    """Edge-aware loss using image gradients"""

    def __init__(self):
        super(EANetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def edge_consistency_loss(self, pred_frame, gt_frame):
        """Edge consistency loss using image gradients"""
        # Compute gradients (edge information)
        pred_grad_x = torch.abs(pred_frame[:, :, :, :-1] - pred_frame[:, :, :, 1:])
        pred_grad_y = torch.abs(pred_frame[:, :, :-1, :] - pred_frame[:, :, 1:, :])

        gt_grad_x = torch.abs(gt_frame[:, :, :, :-1] - gt_frame[:, :, :, 1:])
        gt_grad_y = torch.abs(gt_frame[:, :, :-1, :] - gt_frame[:, :, 1:, :])

        # Edge preservation loss
        edge_loss = self.l1_loss(pred_grad_x, gt_grad_x) + self.l1_loss(pred_grad_y, gt_grad_y)
        return edge_loss

    def forward(self, pred_frame, gt_frame):
        """Returns edge loss"""
        return self.edge_consistency_loss(pred_frame, gt_frame)


class StructuralLoss(nn.Module):
    """Structure similarity loss using SSIM-like comparison"""

    def __init__(self):
        super(StructuralLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def structural_similarity_loss(self, pred_frame, gt_frame):
        """Structural similarity loss based on local statistics"""
        # Create sliding windows for local comparison
        kernel_size = 11
        padding = kernel_size // 2

        # Unfold to create patches
        pred_patches = torch.nn.functional.unfold(pred_frame, kernel_size, padding=padding, stride=1)
        gt_patches = torch.nn.functional.unfold(gt_frame, kernel_size, padding=padding, stride=1)

        # Reshape patches: [B, C*kernel*kernel, H*W] -> [B, H*W, C*kernel*kernel]
        pred_patches = pred_patches.transpose(1, 2)
        gt_patches = gt_patches.transpose(1, 2)

        # Compute local means
        pred_mean = pred_patches.mean(dim=2, keepdim=True)
        gt_mean = gt_patches.mean(dim=2, keepdim=True)

        # Compute local variances
        pred_var = pred_patches.var(dim=2, keepdim=True)
        gt_var = gt_patches.var(dim=2, keepdim=True)

        # Compute covariance
        pred_centered = pred_patches - pred_mean
        gt_centered = gt_patches - gt_mean
        covariance = (pred_centered * gt_centered).mean(dim=2, keepdim=True)

        # SSIM-like structural loss
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (2 * pred_mean * gt_mean + c1) * (2 * covariance + c2)
        denominator = (pred_mean ** 2 + gt_mean ** 2 + c1) * (pred_var + gt_var + c2)

        ssim_map = numerator / (denominator + 1e-8)
        structure_loss = 1 - ssim_map.mean()

        return structure_loss

    def forward(self, pred_frame, gt_frame):
        """Returns structural loss"""
        return self.structural_similarity_loss(pred_frame, gt_frame)


class loss_function:
    """Enhanced loss function: 0.7*reconstruction + 0.1*perceptual + 0.1*structure + 0.1*edge"""

    def __init__(self, resnet_conv, criterion_reconstruction, criterion_perceptual):
        self.resnet_conv = resnet_conv
        self.criterion_reconstruction = criterion_reconstruction
        self.criterion_perceptual = criterion_perceptual

        # Always use edge and structure losses
        self.ea_loss = EANetLoss()
        self.structure_loss = StructuralLoss()

        print("Loss function initialized with new weighting: 0.7*recon + 0.1*percep + 0.1*struct + 0.1*edge")

    def compute_losses(self, It_pred, It_gt):
        """Compute all four loss components"""
        # Reconstruction loss
        loss_reconstruction = self.criterion_reconstruction(It_pred, It_gt)

        # Perceptual loss
        feat_pred = self.resnet_conv(It_pred)
        feat_gt = self.resnet_conv(It_gt)
        loss_perceptual = self.criterion_perceptual(feat_pred, feat_gt)

        # Structure loss
        loss_structure = self.structure_loss(It_pred, It_gt)

        # Edge loss
        loss_edge = self.ea_loss(It_pred, It_gt)

        return loss_reconstruction, loss_perceptual, loss_structure, loss_edge

    def compute_total_loss(self, loss_reconstruction_list, loss_perceptual_list,
                           loss_structure_list, loss_edge_list):
        """Compute total loss with new weighting"""
        # Average each loss component
        avg_reconstruction = sum(loss_reconstruction_list) / len(loss_reconstruction_list)
        avg_perceptual = sum(loss_perceptual_list) / len(loss_perceptual_list)
        avg_structure = sum(loss_structure_list) / len(loss_structure_list)
        avg_edge = sum(loss_edge_list) / len(loss_edge_list)

        # New weighting: 0.7 + 0.1 + 0.1 + 0.1 = 1.0
        total_loss = (0.7 * avg_reconstruction +
                      0.1 * avg_perceptual +
                      0.1 * avg_structure +
                      0.1 * avg_edge)

        return total_loss