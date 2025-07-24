import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from datasets.vimeo_90K import VimeoSepTuplet
from conf.lossfunc import loss_function
import torch.optim.lr_scheduler as lr_scheduler
from models.model_Griffin.synthesis2 import backWarp
from models.model_Griffin.Unet import UNet
from models.model_4.ea_flow_net import EAFlowPredictionNet
from models.model_4.training_strategy import EATrainingStrategy
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import logging
import datetime
from conf import config2, utils2
from conf.utils2 import save_batch_images, makedirs, save_custom_image
from torch.utils.tensorboard import SummaryWriter
import time

##### Parse CmdLine Arguments #####

args, unparsed = config2.get_args()
print(args)
args.width, args.height = 256, 256
train_data_path = "D:\\KIEN\\Dataset\\vimeo_septuplet"  # Update with your dataset path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.resume = False

###############################
# Create checkpoint directory
# os.makedirs(args.checkpoint_dir, exist_ok=True)



# Dataset and DataLoader
train_feeder = VimeoSepTuplet(args.datasetPath, is_training=True, mode="full")
train_loader = DataLoader(train_feeder, batch_size=args.train_batch_size, shuffle=True)
test_feeder = VimeoSepTuplet(args.datasetPath, is_training=False, mode="full")
test_loader = DataLoader(test_feeder, batch_size=args.test_batch_size, shuffle=False)
print(f"Training dataset size: {len(train_feeder)}, Test dataset size: {len(test_feeder)}")
# Ensure dimensions are divisible by 32 (if required)
w, h = args.width, args.height
w, h = (w // 32) * 32, (h // 32) * 32
# print(f"Original dimensions: {width}x{height}, Adjusted dimensions: {w}x{h}")

if args.modelName == 'base3GAP':
    from models.model_Griffin.flowpred2_griffin import FlowPredictionNet
    model = FlowPredictionNet(in_channels=6, hidden_channels=64, kernel_size=3, W=w, H=h, device=device, output_channels=4).to(device)
    interp = UNet(20, 5).to(device)

elif args.modelName == 'Unet':
    model = UNet(12, 4).to(device)
    interp = UNet(20, 5).to(device)
elif args.modelName == 'EA-Net':  # THÊM DÒNG NÀY
    from models.model_4.ea_flow_net import EAFlowPredictionNet
    model = EAFlowPredictionNet(
        in_channels=6, hidden_channels=64, kernel_size=3,
        W=w, H=h, device=device, output_channels=4,
        edge_method='augmentation'
    ).to(device)
    interp = UNet(20, 5).to(device)
    print("\n=== EA-Net ===")
    # frozen_count = 0
    # trainable_count = 0
    #
    # for name, param in model.named_parameters():
    #     if 'down' in name:  # encoder layers
    #         if not param.requires_grad:
    #             frozen_count += 1
    #         else:
    # #             print(f"⚠️  WARNING: {name} should be frozen but is trainable!")
    # #     elif param.requires_grad:
    # #         trainable_count += 1
    #
    # print(f"✅ Encoder layers frozen: {frozen_count}")
    # print(f"✅ Trainable parameters: {trainable_count}")
    # print("✅ Solution 1 implemented correctly - Encoder frozen!\n")

# models = FlowPredictionNet(in_channels=6, hidden_channels=64, kernel_size=3, W=w, H=h, device=device).to(device)
backWarp = backWarp(w, h, device).to(device)
# backWarp_test = backWarp(w, h, device).to(device)
print('# of parameters: %d' % (sum(p.numel() for p in model.parameters()) + sum(k.numel() for k in interp.parameters())))
# Perceptual loss models (ResNet-18)
resnet = resnet18(pretrained=True)
resnet_conv = nn.Sequential(*list(resnet.children())[:-2]).to(device)  # Use layers up to the second last block
for param in resnet_conv.parameters():
    param.requires_grad = False  # Freeze ResNet parameters

# Optimizer
args.scheduler = 'MultiStepLR'
args.opt_name = 'ADAM'

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(interp.parameters()),
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
    weight_decay=args.weight_decay
)

scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 20, 80], gamma=0.1)


# Loss functions
loss_func = loss_function(
    resnet_conv,
    nn.L1Loss().to(device),
    nn.MSELoss().to(device)
)

# Metrics
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


def analyze_regional_performance(model, test_loader, device, epoch):
    """Analyze performance on smooth vs edge-heavy regions"""
    if epoch % 5 != 0:  # Only analyze every 5 epochs
        return

    print(f"\n=== Regional Analysis - Epoch {epoch} ===")
    model.eval()

    smooth_psnr = []
    edge_psnr = []

    with torch.no_grad():
        for i, (inputs, groundTrue, _) in enumerate(test_loader):
            if i >= 10:  # Analyze first 10 batches only
                break

            I1, I2, I3, I4 = [img.to(device) for img in inputs]
            gt = groundTrue[1].to(device)  # Middle frame

            # Get edge density
            edges = model.edge_extractor(I2) if hasattr(model, 'edge_extractor') else None
            if edges is not None:
                edge_density = edges.mean(dim=[2, 3])  # Average per image

                # Forward pass (simplified)
                I1_2 = torch.cat((I1, I2), dim=1)
                I3_4 = torch.cat((I3, I4), dim=1)
                flow_out = model(I1_2, I3_4)
                # ... (simplified synthesis for analysis)

                # Categorize by edge density
                for j in range(len(edge_density)):
                    density = edge_density[j].mean().item()
                    # psnr_val = compute_psnr(predicted[j], gt[j])  # You need to implement this

                    if density < 0.1:
                        # smooth_psnr.append(psnr_val)
                        pass
                    elif density > 0.3:
                        # edge_psnr.append(psnr_val)
                        pass

    # Report results
    if smooth_psnr and edge_psnr:
        print(f"Smooth regions PSNR: {np.mean(smooth_psnr):.2f}")
        print(f"Edge regions PSNR: {np.mean(edge_psnr):.2f}")
        improvement = np.mean(edge_psnr) - np.mean(smooth_psnr)
        print(f"Edge vs Smooth improvement: {improvement:+.2f} dB")

    print("=" * 40)

# Training loop
def train(args, epoch):
    model.train()

    epoch_loss = 0
    epoch_psnr = 0
    epoch_ssim = 0
    num_batches = 0
    for i, (inputs, groundTrue, edges) in enumerate(train_loader):
        # Prepare input frames
        batch_start = time.time()

        ## 1. Loading & preprocessing input
        load_start = time.time()
        I1, I2, I3, I4 = [img.to(device) for img in inputs]  # Input sequence
        gts = [gt.to(device) for gt in groundTrue]  # Ground truth intermediate frames
        start = time.time()
        edge1, edge2, edge3, edge4 = [edge.to(device) for edge in edges]  # add edge
        print(f"🕒 Data load/preprocess: {time.time() - load_start:.3f}s")

        optimizer.zero_grad()

        I1_2 = torch.cat((I1, I2), dim=1)  # Concatenate along the channel dimension
        I3_4 = torch.cat((I3, I4), dim=1)

# ####### EDGE VERSION##############
        edge_I1_2 = torch.cat([edge1, edge2], dim=1)
        edge_I3_4 = torch.cat([edge3, edge4], dim=1)
        # Forward pass for input pairs
        ## 2. Forward model
        model_start = time.time()
        flow_out = model(I1_2, I3_4, edge_I1_2, edge_I3_4)
        print(f"🕒 Model forward: {time.time() - model_start:.3f}s")
####################################

######## NORMAL VERSION ####
        # flow_out = model(I1_2, I3_4)
###########################

        ## 3. Flow splitting
        flow_split_start = time.time()
        F_0_1 = flow_out[:, :2]
        F_1_0 = flow_out[:, 2:]
        print(f"🕒 Flow split: {time.time() - flow_split_start:.3f}s")

        frame_loss_start = time.time()
        # Initialize lists for all 4 loss components  # CHANGED
        loss_reconstruction_list = []
        loss_perceptual_list = []
        loss_structure_list = []  # NEW
        loss_edge_list = []  # NEW

        batch_psnr = 0
        batch_ssim = 0
        # Interpolate each intermediate frame
        for t_idx, It_gt in enumerate(gts):
            frame_start = time.time()
            t = (t_idx + 1) / (len(gts) + 1)

            # Simplified coefficient calculation
            C00 = -t * (1 - t)
            C01 = t * t
            C10 = (1 - t) * (1 - t)
            C11 = -t * (1 - t)

            # Compute intermediate flows
            F_t_0 = C00 * F_0_1 + C01 * F_1_0
            F_t_1 = C10 * F_0_1 + C11 * F_1_0

            # Version 2: backwarp with I1 and I4
            g_I0_F_t_0 = backWarp(I1, F_t_0)
            g_I1_F_t_1 = backWarp(I4, F_t_1)

            ##compute Visionbility Map
            iy = torch.cat((I1, I4, F_0_1, F_1_0, F_t_0, F_t_1, g_I0_F_t_0, g_I1_F_t_1), dim=1)
            io = interp(iy)

            ft0f = io[:, :2, :, :] + F_t_0
            ft1f = io[:, 2:4, :, :] + F_t_1
            vt0 = torch.sigmoid(io[:, 4:5, :, :])
            vt1 = 1 - vt0

            g_I0_F_t_0_f = backWarp(I1, ft0f)
            g_I1_F_t_1_f = backWarp(I4, ft1f)

            # Final interpolation using temporal coefficients
            co_eff = [1 - t, t]
            Ft_p = (co_eff[0] * vt0 * g_I0_F_t_0_f + co_eff[1] * vt1 * g_I1_F_t_1_f) / \
                   (co_eff[0] * vt0 + co_eff[1] * vt1)

            # Compute losses with 4 components  # CHANGED
            L1, L2, L3, L4 = loss_func.compute_losses(Ft_p, It_gt)
            # Append all losses to respective lists
            loss_reconstruction_list.append(L1)
            loss_perceptual_list.append(L2)
            loss_structure_list.append(L3)  # NEW
            loss_edge_list.append(L4)  # NEW

            # Calculate PSNR and SSIM
            psnr_score = psnr(Ft_p, It_gt)
            ssim_score = ssim(Ft_p, It_gt)
            batch_psnr += psnr_score.item()
            batch_ssim += ssim_score.item()
            print(f"   🕒 Frame {t_idx} total: {time.time() - frame_start:.3f}s")

        # Compute total average loss with 4 components  # CHANGED
        avg_total_loss = loss_func.compute_total_loss(
            loss_reconstruction_list,
            loss_perceptual_list,
            loss_structure_list,  # NEW
            loss_edge_list  # NEW
        )
        avg_total_loss.backward()
        optimizer.zero_grad()
        ## 5. Optimizer step
        optim_start = time.time()
        # optimizer.zero_grad()
        optimizer.step()
        print(f"🕒 Optimizer step: {time.time() - optim_start:.3f}s")

        print(f"🕒 Total frame loss calc: {time.time() - frame_loss_start:.3f}s")



        avg_psnr = batch_psnr / len(gts)
        avg_ssim = batch_ssim / len(gts)
        num_batches += 1

        # Track loss for the epoch
        epoch_loss += avg_total_loss.item()
        epoch_psnr += avg_psnr
        epoch_ssim += avg_ssim

        logging.info(
            f'Epoch [{epoch + 1}/{args.max_epoch}], Step [{i + 1}/{len(train_loader)}], Batch Loss: {avg_total_loss.item():.4f}, Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.3f}')
        print(f"✅ Batch {i} done. Total time: {time.time() - batch_start:.3f}s")

    ##### BATCH ENDPOINT #####
    # Average metrics over the epoch
    epoch_psnr /= num_batches
    epoch_ssim /= num_batches

    # Trong training loop - track edge component learning
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: edge_weight = {model.edge_weight.item():.3f}")

        # Check if edge processing is actually helping
        with torch.no_grad():
            E0 = model.edge_extractor(I1)
            edge_density = E0.mean().item()
            print(f"Average edge density: {edge_density:.3f}")
    logging.warning(
        f"Training Epoch [{epoch + 1}/{args.max_epoch}] completed. Average Loss: {epoch_loss / len(train_loader):.4f}, Average PSNR: {epoch_psnr:.2f}, Average SSIM: {epoch_ssim:.3f}")
    return epoch_loss / len(train_loader), epoch_psnr
    ####### EPOCH ENDPOINT #####

#Test loop
def test(args, epoch):
    model.eval()

    test_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_samples = 0
    temp = 0
    args.out_counter = 0
    threshold_psnr = 34.0
    psnr_indices = []

    with torch.no_grad():
        for i, (inputs, groundTrue, edges, path) in enumerate(test_loader):
            # Prepare input frames similar to training
            I1, I2, I3, I4 = [img.to(device) for img in inputs]
            gts = [gt.to(device) for gt in groundTrue]

            edge1, edge2, edge3, edge4 = [edge.to(device) for edge in edges] #add edge
            # Similar to training, concatenate frames along the channel dimension
            I1_2 = torch.cat((I1, I2), dim=1)
            I3_4 = torch.cat((I3, I4), dim=1)
            edge_I1_2 = torch.cat([edge1, edge2], dim=1)
            edge_I3_4 = torch.cat([edge3, edge4], dim=1)
            start = time.time()
            # Forward pass
            # flow_out = model(I1_2, I3_4)
            flow_out = model(I1_2, I3_4, edge_I1_2, edge_I3_4)

            F_0_1 = flow_out[:, :2, :, :]
            F_1_0 = flow_out[:, 2:, :, :]

            # Loss lists for 4 components  # CHANGED
            loss_reconstruction_list = []
            loss_perceptual_list = []
            loss_structure_list = []  # NEW
            loss_edge_list = []  # NEW

            start = time.time()

            for t_idx, It_gt in enumerate(gts):
                t = (t_idx + 1) / (len(gts) + 1)

                # Simplified coefficient calculation
                C00 = -t * (1 - t)
                C01 = t * t
                C10 = (1 - t) * (1 - t)
                C11 = -t * (1 - t)

                # Compute intermediate flows
                F_t_0 = C00 * F_0_1 + C01 * F_1_0
                F_t_1 = C10 * F_0_1 + C11 * F_1_0

                # Version 2: backwarp with I1 and I4
                g_I0_F_t_0 = backWarp(I1, F_t_0)
                g_I1_F_t_1 = backWarp(I4, F_t_1)

                ##compute Visionbility Map
                iy = torch.cat((I1, I4, F_0_1, F_1_0, F_t_0, F_t_1, g_I0_F_t_0, g_I1_F_t_1), dim=1)
                io = interp(iy)

                ft0f = io[:, :2, :, :] + F_t_0
                ft1f = io[:, 2:4, :, :] + F_t_1
                vt0 = torch.sigmoid(io[:, 4:5, :, :])
                vt1 = 1 - vt0

                g_I0_F_t_0_f = backWarp(I1, ft0f)
                g_I1_F_t_1_f = backWarp(I4, ft1f)

                # Final interpolation using temporal coefficients
                co_eff = [1 - t, t]
                Ft_p = (co_eff[0] * vt0 * g_I0_F_t_0_f + co_eff[1] * vt1 * g_I1_F_t_1_f) / \
                       (co_eff[0] * vt0 + co_eff[1] * vt1)

                # Compute losses with 4 components  # CHANGED
                L1, L2, L3, L4 = loss_func.compute_losses(Ft_p, It_gt)

                loss_reconstruction_list.append(L1)
                loss_perceptual_list.append(L2)
                loss_structure_list.append(L3)  # NEW
                loss_edge_list.append(L4)  # NEW

                # Calculate PSNR and SSIM
                psnr_score = psnr(Ft_p, It_gt)
                ssim_score = ssim(Ft_p, It_gt)
                total_psnr += psnr_score.item()
                total_ssim += ssim_score.item()

                if psnr_score.item() > 34 and psnr_score.item() < 35 and int(ssim_score.item()) > 0.87 and int(
                        ssim_score.item()) < 0.93:
                    temp = 1
                    save_batch_images(args, Ft_p, It_gt, t_idx, args.test_epoch_path)
                    logging.info(
                        f"Saved output image for test batch {i} / {len(test_loader)}, PSNR: {psnr_score:.3f} SSIM: {ssim_score:.3f}")
            logging.info(f"Time taken for all frames: {time.time() - start}s")

            if temp: psnr_indices.append(path)
            temp = 0

            # Compute total loss with 4 components  # CHANGED
            total_loss = loss_func.compute_total_loss(
                loss_reconstruction_list,
                loss_perceptual_list,
                loss_structure_list,  # NEW
                loss_edge_list  # NEW
            )
            test_loss += total_loss.item()
            num_samples += 1

    with open("psnr_indices.txt", "w") as f:
        for path in psnr_indices:
            f.write(f"{path}\n")

    # Output average loss and metrics
    avg_test_loss = test_loss / num_samples
    avg_psnr = total_psnr / (num_samples * 3)
    avg_ssim = total_ssim / (num_samples * 3)
    logging.warning(f"Test Loss: {avg_test_loss:.4f}, Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.3f}")
    return avg_test_loss, avg_psnr


def custom(args):
    model.eval()

    with torch.no_grad():

        for i, (inputs, groundTrue) in enumerate(test_loader):
            # Prepare input frames similar to training
            I1, I2, I3, I4 = [img.to(device) for img in inputs]  # Input sequence
            gts = [gt.to(device) for gt in groundTrue]  # Ground truth intermediate frames

            # Similar to training, concatenate frames along the channel dimension
            I1_2 = torch.cat((I1, I2), dim=1)
            I3_4 = torch.cat((I3, I4), dim=1)

            # I_1234 = torch.cat([I1_2, I3_4], dim=1)  # Shape: (batch_size, time, channels, height, width) #Unet
            start = time.time()
            # Forward pass
            flow_out = model(I1_2, I3_4)  # Predict F_0_1 and F_1_0
            # flow_out = models(I_1234)  # Predict F_0_1 and F_1_0 #Unet
            F_0_1 = flow_out[:, :2, :, :]
            F_1_0 = flow_out[:, 2:, :, :]
            logging.info(f"Time taken for flow: {time.time() - start}s")

            start = time.time()
            for i in range(1, args.scale):
                t = i / args.scale

                # Simplified coefficient calculation
                C00 = -t * (1 - t)
                C01 = t * t
                C10 = (1 - t) * (1 - t)
                C11 = -t * (1 - t)

                # Compute intermediate flows
                F_t_0 = C00 * F_0_1 + C01 * F_1_0
                F_t_1 = C10 * F_0_1 + C11 * F_1_0

                # Version 2: backwarp with I1 and I4
                g_I0_F_t_0 = backWarp(I1, F_t_0)
                g_I1_F_t_1 = backWarp(I4, F_t_1)

                ##compute Visionbility Map
                iy = torch.cat((I1, I4, F_0_1, F_1_0, F_t_0, F_t_1, g_I0_F_t_0, g_I1_F_t_1), dim=1)
                #   3 + 3 + 2 +   2 +  2 +  2    + 3   + 3 = 20
                io = interp(iy)

                ft0f = io[:, :2, :, :] + F_t_0
                ft1f = io[:, 2:4, :, :] + F_t_1
                vt0 = torch.sigmoid(io[:, 4:5, :, :])
                vt1 = 1 - vt0

                g_I0_F_t_0_f = backWarp(I1, ft0f)
                g_I1_F_t_1_f = backWarp(I4, ft1f)

                # Final interpolation using temporal coefficients
                co_eff = [1 - t, t]
                Ft_p = (co_eff[0] * vt0 * g_I0_F_t_0_f + co_eff[1] * vt1 * g_I1_F_t_1_f) / \
                       (co_eff[0] * vt0 + co_eff[1] * vt1)
                # Ft_p = ((1 - t) * g_I0_F_t_0 + t * g_I1_F_t_1)
                # print(f"shape of Ft_p: {Ft_p.shape}")
                # Ft_p: torch.Size([16, 3, 256, 256])
                logging.info(f"Time taken for frame {i}: {time.time() - start}s")
                save_custom_image(args, Ft_p, i, args.test_epoch_path)

            logging.info(f"Time taken for all frames: {time.time() - start}s")

            # logging.info(f"Time taken for 1 time 4in3out interpolation: {time.time() - start}s")
            # Compute total loss for the batch

def ssim_test(args):
    model.eval()

    total_psnr = 0
    total_ssim = 0
    num_samples = 0
    args.out_counter = 0
    psnr_threshold = 30
    psnr_indices = []

    with torch.no_grad():
        for i, (inputs, groundTrue) in enumerate(test_loader):
            # Prepare input frames similar to training
            I1, I2, I3, I4 = [img.to(device) for img in inputs]
            gts = [gt.to(device) for gt in groundTrue]

            # Similar to training, concatenate frames along the channel dimension
            I1_2 = torch.cat((I1, I2), dim=1)
            I3_4 = torch.cat((I3, I4), dim=1)

            # Forward pass
            flow_out = model(I1_2, I3_4)
            F_0_1 = flow_out[:, :2, :, :]
            F_1_0 = flow_out[:, 2:, :, :]

            # Loss lists (even if not used for final calculation)  # CHANGED
            loss_reconstruction_list = []
            loss_perceptual_list = []
            loss_structure_list = []    # NEW
            loss_edge_list = []         # NEW

            for t_idx, It_gt in enumerate(gts):
                t = (t_idx + 1) / (len(gts) + 1)

                # Compute intermediate flows
                C00 = -t * (1 - t)
                C01 = t * t
                C10 = (1 - t) * (1 - t)
                C11 = -t * (1 - t)

                F_t_0 = C00 * F_0_1 + C01 * F_1_0
                F_t_1 = C10 * F_0_1 + C11 * F_1_0

                g_I0_F_t_0 = backWarp(I1, F_t_0)
                g_I1_F_t_1 = backWarp(I4, F_t_1)

                iy = torch.cat((I1, I4, F_0_1, F_1_0, F_t_0, F_t_1, g_I0_F_t_0, g_I1_F_t_1), dim=1)
                io = interp(iy)

                ft0f = io[:, :2, :, :] + F_t_0
                ft1f = io[:, 2:4, :, :] + F_t_1
                vt0 = torch.sigmoid(io[:, 4:5, :, :])
                vt1 = 1 - vt0

                g_I0_F_t_0_f = backWarp(I1, ft0f)
                g_I1_F_t_1_f = backWarp(I4, ft1f)

                co_eff = [1 - t, t]
                Ft_p = (co_eff[0] * vt0 * g_I0_F_t_0_f + co_eff[1] * vt1 * g_I1_F_t_1_f) / \
                       (co_eff[0] * vt0 + co_eff[1] * vt1)

                # Compute PSNR and SSIM (main purpose of this function)
                psnr_score = psnr(Ft_p, It_gt)
                ssim_score = ssim(Ft_p, It_gt)
                total_psnr += psnr_score.item()
                total_ssim += ssim_score.item()

                # Save logic unchanged
                if psnr_score.item() > psnr_threshold:
                    save_batch_images(args, Ft_p, It_gt, t_idx, args.test_epoch_path)
                    psnr_indices.append((i, t_idx))

            num_samples += 1

    # Write indices to file
    with open("psnr_indices.txt", "w") as f:
        for idx in psnr_indices:
            f.write(f"{idx[0]} {idx[1]}\n")

    # Output average loss and metrics
    avg_psnr = total_psnr / (num_samples * 3)
    avg_ssim = total_ssim / (num_samples * 3)
    logging.warning(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.3f}")
    return avg_psnr, avg_ssim  # Fixed return values


def main(args):
    utils2.configure_logging2(args)

    if args.test_mode == "test":
        print("*********Testing********")
        unique_id = str(np.random.randint(0, 100000))
        args.uid = unique_id
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        args.test_epoch_path = os.path.join('./Results', args.timestamp + '_' + args.uid)
        makedirs(args.test_epoch_path)

        test_loss, test_psnr = test(args, args.start_epoch)
        logging.warning(f"Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.2f}")
        return
    elif args.test_mode == "custom":
        print("*********Custom Testing********")
        unique_id = str(np.random.randint(0, 100000))
        args.uid = unique_id
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        args.test_epoch_path = os.path.join('./Results', args.timestamp + '_' + args.uid)
        makedirs(args.test_epoch_path)
        custom(args)
        return
    elif args.test_mode == "ssim_test":
        print("*********SSIM Testing********")
        unique_id = str(np.random.randint(0, 100000))
        args.uid = unique_id
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        args.test_epoch_path = os.path.join('./Results', args.timestamp + '_' + args.uid)
        makedirs(args.test_epoch_path)
        test_psnr, test_ssim = ssim_test(args)
        logging.warning(f"Test SSIM: {test_ssim:.4f}, Test PSNR: {test_psnr:.2f}")
        

        return


    if not args.resume:
        print("*********Starting New********")
        unique_id = str(np.random.randint(0, 100000))
        args.uid = unique_id
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        args.save_path = os.path.join('./Results', args.timestamp + '_' + args.uid)
    else:
        print("*********Resuming********")
        # args.save_path = os.path.join('./Results', args.timestamp + '_' + args.uid) #parse save path is the same as the previous one id
        utils2.load_checkpoint(args, args.checkpoint_dir, model, interp, optimizer, scheduler) #parse args.checkpoint_dir
        if args.save_path == '':
            unique_id = str(np.random.randint(0, 100000))
            args.uid = unique_id
            args.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            args.save_path = os.path.join('./Results', args.timestamp + '_' + args.uid)

    makedirs(args.save_path)



    # args.save_checkpoint_dir = os.path.join(args.save_path, 'checkpoints')
    # os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.warning(f"""models Name: {args.modelName}
    batch_size: {args.train_batch_size}
    val_batch_size: {args.val_batch_size}
    dataset: {args.datasetName}
    img_h: {args.height}
    img_w: {args.width}
    train_num: {len(train_feeder)}
    val_num: {len(test_feeder)}
    number_workers: {args.num_workers}
    epochs: {args.max_epoch}
    loss: {args.loss}
    lr: {args.lr}
    optimizer: {args.opt_name}
    weight_decay: {args.weight_decay}
    scheduler: {args.scheduler}
    factor: {args.factor}
    patience: {args.patience}
    min_lr: {args.min_lr}
    models parameters: {str(utils2.count_network_parameters(model) + utils2.count_network_parameters(interp))}""")
    writer = SummaryWriter(log_dir='./logs')

    for epoch in range(args.start_epoch, args.max_epoch):
        logging.info(f"Starting Epoch {epoch + 1}/{args.max_epoch}")
        train_loss, train_psnr = train(args, epoch)
        # utils2.check_gradient(models)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'best_psnr': args.best_psnr,
            'state_dict': model.state_dict(),
            'interp_state_dict': interp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
            'scheduler': scheduler.state_dict(),
        }
        utils2.save_checkpoint(args, checkpoint, is_best=False, epoch=epoch)

        # Run validating
        print('Run Validating')  # Check bug
        args.test_epoch_path = os.path.join(args.save_path, "Result_Images", f'epoch_{epoch}')
        makedirs(args.test_epoch_path)

        val_loss, val_psnr = test(args, epoch)
        ####################Set the writer####################
        writer.add_scalar('Training Loss', train_loss, epoch+1)
        writer.add_scalar('Training PSNR', train_psnr, epoch+1)
        writer.add_scalar('Validation Loss', val_loss, epoch+1)
        writer.add_scalar('Validation PSNR', val_psnr, epoch+1)



        # Log the epoch results#################################
        logging.warning(f"Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best models
        print('Save best models')  # Check bug 1029
        is_best = val_psnr > args.best_psnr
        if is_best:
            args.best_psnr = max(val_psnr, args.best_psnr)
            utils2.save_checkpoint(args, checkpoint, is_best=is_best, epoch=epoch)
            print("Best Weights updated for decreased psnr\n")
        else:
            print("Weights Not updated for undecreased psnr\n")

        # schedule the learning rate
        scheduler.step(val_loss)

    writer.close()

    print("*********Finish Training********")
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    main(args)
