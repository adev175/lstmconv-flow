import os
import shutil
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

import torch
from torchvision import transforms
from torchvision.utils import save_image

# Configure logging
def configure_logging1(args):
    '''
    Configure logging all for the training
    '''
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=os.path.join(args.save_path, 'training.log'),
                        level=logging.INFO,
                        format=log_format,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def configure_logging2(args):
    '''
    Configure 2 logging, 1 for console >debug and 1 for writing to file >info
    '''

    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Remove any default handlers if present
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create logger
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)  # Set lowest level to capture everything at the handler level

    # Define formatter
    formatter = logging.Formatter(log_format)

    # Console handler for DEBUG and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler for INFO and above
    file_handler = logging.FileHandler(os.path.join(args.save_path, 'training.log'), mode='w')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(args, model, interp, optimizer, scheduler, epoch, is_best=False):
    """
    Save checkpoint with model, interpolation network, optimizer, and scheduler.
    Includes training metadata (epoch, best_psnr, etc.)
    """
    save_dir = args.save_path if hasattr(args, 'save_path') else './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'best_psnr': args.best_psnr if hasattr(args, 'best_psnr') else 0.0,
        'state_dict': model.state_dict(),
        'interp_state_dict': interp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'lr': args.lr if hasattr(args, 'lr') else 1e-4
    }

    filename = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save(checkpoint, filename)
    print(f"💾 Saved checkpoint to {filename}")

    if is_best:
        best_path = os.path.join(save_dir, "best.pth")
        torch.save(checkpoint, best_path)
        print("🏅 Best model updated!")

def load_checkpoint(args, checkpoint_path, model, interp, optimizer, scheduler):
    """
    Load compatible weights from checkpoint. Ignore any shape mismatches.
    No parameter freezing logic included.
    """
    print(f"\n🔄 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_state = checkpoint['state_dict']
    model_state = model.state_dict()

    compatible_params = {}
    incompatible = []
    new_params = []

    for name, param in checkpoint_state.items():
        if name in model_state and model_state[name].shape == param.shape:
            compatible_params[name] = param
        else:
            incompatible.append(name)

    for name in model_state:
        if name not in checkpoint_state:
            new_params.append(name)

    print(f"✅ Loaded {len(compatible_params)} / {len(model_state)} parameters")
    if incompatible:
        print("⚠️ Skipped incompatible parameters:")
        for name in incompatible:
            print(f"   - {name}")

    if new_params:
        print("🆕 New parameters randomly initialized:")
        for name in new_params:
            print(f"   - {name}")

    model.load_state_dict(compatible_params, strict=False)

    # Load interpolation and optimizer
    try:
        interp.load_state_dict(checkpoint['interp_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    except:
        print("⚠️ Failed to load optimizer/scheduler — reinitializing.")

    args.start_epoch = checkpoint.get('epoch', 0)
    args.best_psnr = checkpoint.get('best_psnr', 0.0)
    args.lr = checkpoint.get('lr', 1e-4)

    return model, optimizer, scheduler


def count_network_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


##########################
# ETC
##########################

def makedirs(path):
    if not os.path.exists(path):
        # print("[*] Make directories: {}".format(path))
        os.makedirs(path)  # os.makedirs: creates all the intermediate directories if they don't exist


def remove_file(path):
    if os.path.exists(path):
        # print("[*] Removed: {}".format(path))
        os.remove(path)


def count_network_parameters(model):
    # Calculate models parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in parameters])

    return N


# Tensorboard
def log_tensorboard(writer, losses, psnr, ssim, lr, epoch, mode='train'):
    for k, v in losses.items():
        writer.add_scalar('Loss/%s/%s' % (k, mode), v.avg, epoch)
    writer.add_scalar('PSNR/%s' % mode, psnr, epoch)
    writer.add_scalar('SSIM/%s' % mode, ssim, epoch)
    if mode == 'train':
        writer.add_scalar('lr', lr, epoch)


def quantize(img, rgb_range=255.):
    return img.mul(255. / rgb_range).round()


def save_image(img, path):
    # img : torch Tensor of size (C, H, W)
    q_im = quantize(img.data.mul(255))
    if len(img.size()) == 2:  # grayscale image
        im = Image.fromarray(q_im.cpu().numpy().astype(np.uint8), 'L')
    elif len(img.size()) == 3:
        im = Image.fromarray(q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
    else:
        pass
    im.save(path)


def save_batch_images(args, ims_pred, ims_gt, frame_idx,  epoch_path):
    # Check if epoch_path exists
    makedirs(epoch_path)  # ['./Results/20220425_1052_8827', 'Result_Images', 'Epoch_0']

    # Save every image in batch to indicated location
    for j in range(ims_pred.size(0)):
        pred_name = "im_" + str(args.out_counter) + '_out' + "_im_" + str(frame_idx + 1) + ".png"  # im_0_out_im_1.png
        gt_name = "im_" + str(args.out_counter) + '_gt' + "_im_" + str(frame_idx + 1) + ".png"  # im_0_gt_im_1.png

        save_image(ims_pred[j, :, :, :], os.path.join(epoch_path, pred_name))
        save_image(ims_gt[j, :, :, :], os.path.join(epoch_path, gt_name))

        # Save reference images at desired size if args.create_reference_images is True and epoch == 0
        """
        if not args.create_reference_images and epoch_path.split('\\')[2].split('_')[1] == 0:
            reference_path = os.path.join(args.save_path, args.reference_folder)
            makedirs(reference_path)

            input1_name = str(args.out_counter) + '_1.png',
            input2_name = str(args.out_counter) + '_2.png'

            save_image(reverse_normalize(input1[j, :, :, :]), os.path.join(reference_path, input1_name))
            save_image(reverse_normalize(input2[j, :, :, :]), os.path.join(reference_path, input2_name))
        """
        args.out_counter += 1  # make it 1-end instead of 1-batch_size

def save_custom_image(args, ims_pred, frame_idx, path):

    makedirs(path)  # ['./Results/20220425_1052_8827', 'Custom_Result_Images']

    # Save every image in batch to indicated location
    for j in range(ims_pred.size(0)):
        pred_name = "im_" + str(args.out_counter) + '_out' + "_im_" + str(frame_idx + 1) + ".png"  # im_0_out_im_1.png

        save_image(ims_pred[j, :, :, :], os.path.join(path, pred_name))

        # Save reference images at desired size if args.create_reference_images is True and epoch == 0
        """
        if not args.create_reference_images and epoch_path.split('\\')[2].split('_')[1] == 0:
            reference_path = os.path.join(args.save_path, args.reference_folder)
            makedirs(reference_path)

            input1_name = str(args.out_counter) + '_1.png',
            input2_name = str(args.out_counter) + '_2.png'

            save_image(reverse_normalize(input1[j, :, :, :]), os.path.join(reference_path, input1_name))
            save_image(reverse_normalize(input2[j, :, :, :]), os.path.join(reference_path, input2_name))
        """
        args.out_counter += 1  # make it 1-end instead of 1-batch_size

def check_gradient(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(f"{name}, Gradient Norm: {param.grad.norm()}")
            else:
                print(f"{name} has no gradient")

def advance_check_gradient(model, writer, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_data = param.grad.cpu().detach()
            if torch.isnan(grad_data).any() or torch.isinf(grad_data).any():
                logging.warning(f"Skipping histogram for {name} due to NaN or Inf values in gradients.")
                continue

            # Safe conversion to numpy array and then to float
            grad_numpy = grad_data.numpy().astype(np.float32)  # Use np.float32 to ensure corequires_gradmpatibility

            # Log the histogram
            writer.add_histogram(f'Gradients/{name}', grad_numpy, epoch)
        else:
            logging.warning(f"No gradient for {name}, skipping histogram logging.")

import cv2
import os

def images_2_video(folder_path, output_fps=None):
    # Get all image files sorted
    images = sorted(
        [img for img in os.listdir(folder_path) if img.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(x.split('(')[-1].split(')')[0])
    )
    num_images = len(images)

    if num_images == 0:
        print("No images found in the folder.")
        return

    # Read the first image to get frame size
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    size = (width, height)

    # Set output video path
    output_path = os.path.join(folder_path, "output_video.mp4")

    # Use provided FPS or default to number of images
    fps = output_fps if output_fps is not None else num_images

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Video saved at {output_path} with {fps} FPS.")

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Video saved at {output_path} with {fps} FPS.")


