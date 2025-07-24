# === Phase 1: Benchmark loss component timings ===
import torch
import torch.nn.functional as F
import numpy as np
import time
from lossfunc import StructuralLoss, EANetLoss


def benchmark_loss_components(pred_frame, gt_frame, resnet_conv=None, iterations=100):
    """
    Benchmark time cost of each loss component: L1, perceptual, structure, edge
    Return: dict of {loss_name: (mean_time, std_time)}
    """
    timings = {
        'reconstruction': [],
        'perceptual': [],
        'structure': [],
        'edge': []
    }

    # Set eval mode and disable grad
    resnet_conv.eval()
    with torch.no_grad():
        for _ in range(iterations):
            torch.cuda.synchronize()

            # Reconstruction loss (L1)
            t0 = time.time()
            _ = F.l1_loss(pred_frame, gt_frame)
            torch.cuda.synchronize()
            timings['reconstruction'].append(time.time() - t0)

            # Perceptual loss (ResNet18)
            t0 = time.time()
            _ = resnet_conv(pred_frame)
            _ = resnet_conv(gt_frame)
            torch.cuda.synchronize()
            timings['perceptual'].append(time.time() - t0)

            # Structure loss (SSIM-like)
            t0 = time.time()
            _ = StructuralLoss()(pred_frame, gt_frame)
            torch.cuda.synchronize()
            timings['structure'].append(time.time() - t0)

            # Edge consistency loss
            t0 = time.time()
            _ = EANetLoss()(pred_frame, gt_frame)
            torch.cuda.synchronize()
            timings['edge'].append(time.time() - t0)

    # Aggregate results
    result = {}
    for name, records in timings.items():
        result[name] = (np.mean(records), np.std(records))
    return result


if __name__ == '__main__':
    # Dummy tensors (simulate output and ground truth)
    B, C, H, W = 8, 3, 256, 256
    pred = torch.rand((B, C, H, W)).cuda()
    gt = torch.rand((B, C, H, W)).cuda()

    from torchvision.models import resnet18
    resnet = resnet18(pretrained=True).cuda()
    resnet_conv = torch.nn.Sequential(*list(resnet.children())[:-2]).cuda()
    for p in resnet_conv.parameters():
        p.requires_grad = False

    # Run benchmark
    results = benchmark_loss_components(pred, gt, resnet_conv=resnet_conv, iterations=50)
    print("\n=== Loss Timing Results (mean ± std in seconds) ===")
    for name, (mean_t, std_t) in results.items():
        print(f"{name:>15}: {mean_t:.4f} ± {std_t:.4f} s")

    """
    === Loss Timing Results (mean ± std in seconds) ===
 reconstruction: 0.0278 ± 0.1942 s
     perceptual: 0.0416 ± 0.2375 s  <------ bottle neck
      structure: 0.0189 ± 0.0283 s
           edge: 0.0006 ± 0.0005 s   <------ very fast

    """


