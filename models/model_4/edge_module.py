# models/model_4/edge_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class CannyEdgeExtractor(nn.Module):
    """Simple Canny edge extraction for EA-Net"""

    def __init__(self, low_threshold=50, high_threshold=150):
        super(CannyEdgeExtractor, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, images):
        """
        Extract edges from input images
        Args:
            images: (B, C, H, W) tensor
        Returns:
            edges: (B, C, H, W) tensor with edge maps
        """
        batch_size, channels, height, width = images.shape
        edge_maps = torch.zeros_like(images)

        for b in range(batch_size):
            for c in range(channels):
                # Convert to numpy and apply Canny
                img_np = images[b, c].detach().cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)

                # Canny edge detection
                edges = cv2.Canny(img_np, self.low_threshold, self.high_threshold)
                edges = edges.astype(np.float32) / 255.0

                edge_maps[b, c] = torch.from_numpy(edges).to(images.device)

        return edge_maps

