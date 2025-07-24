import os
import click
import cv2
import torch
from PIL import Image
import numpy as np
import models
from torchvision import transforms
from torch.functional import F
import time

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_forward = transforms.ToTensor()
trans_backward = transforms.ToPILImage()
if device != "cpu":
    mean = [0.429, 0.431, 0.397]
    mean_neg = [-m for m in mean]
    std = [1] * 3
    trans_forward = transforms.Compose([trans_forward, transforms.Normalize(mean=mean, std=std)])
    trans_backward = transforms.Compose([transforms.Normalize(mean=mean_neg, std=std), trans_backward])


# from model_2.flowpred2_EDblock import FlowPredictionNet
from model_Griffin.flowpred2_griffin import FlowPredictionNet

from model_2.Unet import UNet
flow = FlowPredictionNet(in_channels=6, hidden_channels=64, kernel_size=3, W=256, H=256, device=device, output_channels=4).to(device)
interp = UNet(20, 5).to(device)



def setup_back_warp(w, h):
    global back_warp
    with torch.set_grad_enabled(False):
        back_warp = flow.backWarp(w, h, device).to(device)

def load_models(checkpoint):
    states = torch.load(checkpoint, map_location='cpu')
    interp.load_state_dict(states['interp_state_dict'])
    flow.load_state_dict(states['state_dict'])

def interpolate_batch(frames, factor):
    frame0 = torch.stack(frames[:-1])
    frame1 = torch.stack(frames[1:])

    i0 = frame0.to(device)
    i1 = frame1.to(device)
    ix = torch.cat([i0, i1], dim=1)
    start = time.time()
    flow_out = flow(ix)
    f01 = flow_out[:, :2, :, :]
    f10 = flow_out[:, 2:, :, :]

    frame_buffer = []
    for i in range(1, factor):
        t = i / factor
        temp = -t * (1 - t)
        co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

        ft0 = co_eff[0] * f01 + co_eff[1] * f10
        ft1 = co_eff[2] * f01 + co_eff[3] * f10

        gi0ft0 = back_warp(i0, ft0)
        gi1ft1 = back_warp(i1, ft1)

        iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
        io = interp(iy)

        ft0f = io[:, :2, :, :] + ft0
        ft1f = io[:, 2:4, :, :] + ft1
        vt0 = torch.sigmoid(io[:, 4:5, :, :])
        vt1 = 1 - vt0

        gi0ft0f = back_warp(i0, ft0f)
        gi1ft1f = back_warp(i1, ft1f)

        co_eff = [1 - t, t]

        ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
               (co_eff[0] * vt0 + co_eff[1] * vt1)

        frame_buffer.append(ft_p)
    t_out = time.time() - start
    print(f"Time taken for interpolation: {t_out}s")

    # print(f"Generated {len(frame_buffer)} intermediate frames for factor {factor}")
    # # In ra thông tin chi tiết của frame_buffer
    # for idx, frame in enumerate(frame_buffer):
    #     print(f"Frame {idx}: type={type(frame)}, shape={frame.shape}, dtype={frame.dtype}")
    return frame_buffer

def extract_frames_from_video(video_path, output_dir):
    vin = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0
    f_path = os.path.join(output_dir,"f_paths")
    os.makedirs(f_path, exist_ok=True)
    while True:
        ret, frame = vin.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        # Save frame as an image file
        frame.save(os.path.join(f_path, f'frame_{frame_id:04d}.png'))
        frame_id += 1
        frames.append(frame)
    vin.release()
    return frames

def load_frames_from_directory(directory):
    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))])
    frames = [Image.open(img).convert("RGB") for img in image_files]
    return frames

def prepare_frames(frames, w, h):
    prepared_frames = []
    for frame in frames:
        frame = frame.resize((w, h), Image.LANCZOS)
        frame = trans_forward(frame)
        prepared_frames.append(frame)
    return prepared_frames

def process_and_save_frames(frames, output_dir, checkpoint, factor, batch_size):
    # Load the models
    load_models(checkpoint)

    # Setup dimensions
    test_image = frames[0]
    w0, h0 = test_image.size
    w, h = (w0 // 32) * 32, (h0 // 32) * 32  # Ensure dimensions are multiples of 32
    setup_back_warp(w, h)  # Setup the back warp for these dimensions

    # Prepare frames
    frames = prepare_frames(frames, w, h)
    print(len(frames))

    # Prepare for frame processing
    batch = []
    frame_id = 0

    while frames:
        # Load a batch of frames
        batch = frames[:batch_size]
        frames = frames[batch_size:]
        print("Loaded batch")
        if len(batch) <= 1:  # Break if less than two frames to process
            break

        # Process and interpolate batch
        intermediate_frames = interpolate_batch(batch, factor)
        print(f"Number of intermediate frames created: {len(intermediate_frames)}")

        # Save each frame and its interpolated frames
        for i in range(len(batch) - 1):
            # Save original frame
            image_path = f"{output_dir}/frame_{frame_id:04d}.png"
            save_frame(batch[i], image_path)
            frame_id += 1
            print(f'Saved frame {frame_id}')

            # Save interpolated frames
            for j, frame in enumerate(intermediate_frames):
                image_path = f"{output_dir}/frame_{frame_id:04d}_interpolated_{j:02d}.png"
                save_frame(frame[i], image_path)
                frame_id += 1
                print(f'Saved interpolated frame {frame_id}')

        # Save the last frame of the batch
        image_path = f"{output_dir}/frame_{frame_id:04d}.png"
        save_frame(batch[-1], image_path)
        frame_id += 1
        print(f'Saved frame {frame_id}')

        print(f'Processed {len(batch) - 1} frames')  # Progress output

def save_frame(tensor, path):
    """Converts a tensor to a PIL image and saves it to a file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image = trans_backward(tensor)
        image.save(path)
        print(f"Saving frame to {os.path.abspath(path)}")
    except Exception as e:
        print(f"Failed to save frame: {e}")

@click.command('Evaluate Model by converting a low-FPS video or image sequence to high-fps images')
@click.argument('input_path')
@click.option('--input_type', type=click.Choice(['video', 'directory']), help='Type of input: video or directory')
@click.option('--checkpoint', help='Path to models checkpoint')
@click.option('--output_dir', help='Directory to save output images')
@click.option('--batch', default=2, help='Number of frames to process in single forward pass')
@click.option('--scale', default=4, help='Scale Factor of FPS')

def main(input_path, input_type, checkpoint, output_dir, batch, scale):
    print(f"Input path: {input_path}")
    print(f"Input type: {input_type}")
    print(f"Checkpoint path: {checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch}")
    print(f"Scale factor: {scale}")

    if input_type == 'video':
        frames = extract_frames_from_video(input_path, output_dir)
    elif input_type == 'directory':
        frames = load_frames_from_directory(input_path)
    else:
        print("Invalid input type. Please choose 'video' or 'directory'.")
        return

    if not frames:
        print("No frames found. Exiting.")
        return

    process_and_save_frames(frames, output_dir, checkpoint, scale, batch)

if __name__ == '__main__':
    main()
