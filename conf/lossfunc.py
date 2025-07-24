# === Updated loss function with LossConfigManager support ===
import torch
import torch.nn as nn
from .lossconfig import LossConfigManager

class EANetLoss(nn.Module):
    def __init__(self):
        super(EANetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def edge_consistency_loss(self, pred_frame, gt_frame):
        pred_grad_x = torch.abs(pred_frame[:, :, :, :-1] - pred_frame[:, :, :, 1:])
        pred_grad_y = torch.abs(pred_frame[:, :, :-1, :] - pred_frame[:, :, 1:, :])
        gt_grad_x = torch.abs(gt_frame[:, :, :, :-1] - gt_frame[:, :, :, 1:])
        gt_grad_y = torch.abs(gt_frame[:, :, :-1, :] - gt_frame[:, :, 1:, :])
        edge_loss = self.l1_loss(pred_grad_x, gt_grad_x) + self.l1_loss(pred_grad_y, gt_grad_y)
        return edge_loss

    def forward(self, pred_frame, gt_frame):
        return self.edge_consistency_loss(pred_frame, gt_frame)


class StructuralLoss(nn.Module):
    def __init__(self):
        super(StructuralLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def structural_similarity_loss(self, pred_frame, gt_frame):
        kernel_size = 11
        padding = kernel_size // 2
        pred_patches = torch.nn.functional.unfold(pred_frame, kernel_size, padding=padding, stride=1)
        gt_patches = torch.nn.functional.unfold(gt_frame, kernel_size, padding=padding, stride=1)
        pred_patches = pred_patches.transpose(1, 2)
        gt_patches = gt_patches.transpose(1, 2)
        pred_mean = pred_patches.mean(dim=2, keepdim=True)
        gt_mean = gt_patches.mean(dim=2, keepdim=True)
        pred_var = pred_patches.var(dim=2, keepdim=True)
        gt_var = gt_patches.var(dim=2, keepdim=True)
        pred_centered = pred_patches - pred_mean
        gt_centered = gt_patches - gt_mean
        covariance = (pred_centered * gt_centered).mean(dim=2, keepdim=True)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * pred_mean * gt_mean + c1) * (2 * covariance + c2)
        denominator = (pred_mean ** 2 + gt_mean ** 2 + c1) * (pred_var + gt_var + c2)
        ssim_map = numerator / (denominator + 1e-8)
        structure_loss = 1 - ssim_map.mean()
        return structure_loss

    def forward(self, pred_frame, gt_frame):
        return self.structural_similarity_loss(pred_frame, gt_frame)


class loss_function:
    def __init__(self, resnet_conv, criterion_reconstruction, criterion_perceptual, config=None):
        self.resnet_conv = resnet_conv
        self.criterion_reconstruction = criterion_reconstruction
        self.criterion_perceptual = criterion_perceptual
        self.ea_loss = EANetLoss()
        self.structure_loss = StructuralLoss()
        self.config = config or LossConfigManager()  # default: recon=1.0
        print(f"Loss config: {self.config}")

    def compute_losses(self, It_pred, It_gt):
        losses = {
            'recon': self.criterion_reconstruction(It_pred, It_gt),
            'perceptual': self.criterion_perceptual(
                self.resnet_conv(It_pred), self.resnet_conv(It_gt)),
            'structure': self.structure_loss(It_pred, It_gt),
            'edge': self.ea_loss(It_pred, It_gt)
        }
        return losses

    def compute_total_loss(self, loss_dict):
        total = 0.0
        for k in ['recon', 'perceptual', 'structure', 'edge']:
            total += self.config[k] * loss_dict[k]
        return total
