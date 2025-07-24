import datetime
import argparse
import numpy as np


##### import path #####

## Dataset path for Vimeo_90K
# datasetPath = 'D:/KIEN/Dataset/Vimeo_90K/'
datasetPath = 'D:/KIEN/Dataset/vimeo_septuplet/'


arg_lists = []
parser = argparse.ArgumentParser(description='My Network')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

### NETWORK ###
network_arg = add_argument_group('Network')
network_arg.add_argument('--modelName', '-n', type=str, default='base', help='models architecture',
                         choices=["base3GAP", "Unet", "EA-Net"])


# ### PARAMETERS FOR RSTCANET ###
# rstcanet_arg = add_argument_group('RSTCANet')
# rstcanet_arg.add_argument('--dim', type=int, default=192, help='channel number')
# rstcanet_arg.add_argument('--dim_out', type=int, default=128, help='channel number')
# rstcanet_arg.add_argument('--window_size', type=int, default=8, help='window size of Swin Transformer')
# rstcanet_arg.add_argument('--num_head', type=int, default=6, help='number of heads in Swin Transformer')
# rstcanet_arg.add_argument('--num_groups', type=int, default=5, help='number of RSTCAGroup')
# rstcanet_arg.add_argument('--num_blocks', type=int, default=6, help='number of RSTCAB')
# rstcanet_arg.add_argument('--scale_factor', type=int, default=8, help='Scale factor of Pixel Shuffle')


### DATASET ###
dataset_arg = add_argument_group('Dataset')
dataset_arg.add_argument('--datasetName', type=str, default='VimeoSepTuplet',
                         choices=['UCF101', 'Vimeo_90K', 'VimeoSepTuplet', 'Snufilm'])
dataset_arg.add_argument('--datasetPath', default=datasetPath, help='the path of selected datasets')


### DATALOADER ###
dataloader_arg = add_argument_group('Dataloader')
dataloader_arg.add_argument('--train_batch_size', type=int, default=16, help='train batch size')
dataloader_arg.add_argument('--val_batch_size', type=int, default=16, help='val batch size')
dataloader_arg.add_argument('--test_batch_size', type=int, default=16, help='val batch size')
dataloader_arg.add_argument('--num_workers', '-w', type=int, default=4,
                    help='parallel workers for loading training samples')


### TRAIN/TEST PARAMETERS ###
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--loss', type=str, default='1*L1', help='loss function for optimization')
learn_arg.add_argument('--checkpoint_dir', type=str,
                       default=r"D:\M2_final\new_M2\checkpoint\base3GAP_epoch199.pth",
                       help='path to checkpoint dir')
learn_arg.add_argument('--resume', default=False, type=bool, help='resume training from the checkpoint')
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', '-e', type=int, default=251, help='Number of epochs to train')
learn_arg.add_argument('--best_psnr', type=float, default=0.)
learn_arg.add_argument('--save_images', type=bool, default=False)

### OPTIMIZER ###
optimizer_arg = add_argument_group('Optimizer')
optimizer_arg.add_argument('--opt_name', type=str, default='ADAM', help='Optimizer used for training')
optimizer_arg.add_argument('--lr', type=float, default=1e-4, help='learning rate')
optimizer_arg.add_argument('--weight_decay', type=float, default=1e-6, help='the weight decay for whole network ')
optimizer_arg.add_argument('--beta1', type=float, default=0.9)
optimizer_arg.add_argument('--beta2', type=float, default=0.999)
optimizer_arg.add_argument('--eps', type=float, default=1e-8)

### IMAGES SIZE ###
img_arg = add_argument_group('Image Inf')
img_arg.add_argument('--height', default=256, type=int, help='image height')
img_arg.add_argument('--width', default=256, type=int, help='image width')
img_arg.add_argument('--channel', default=3, type=int, help='image channel')


### TRANSFORM ARGUMENTS ###
trans_arg = add_argument_group('Transforms argument')
trans_arg.add_argument('--brightness', default=0.05, help='brightness factor of ColorJitter')
trans_arg.add_argument('--contrast', default=0.05, help='contrast factor of ColorJitter')
trans_arg.add_argument('--saturation', default=0.05, help='saturation factor of ColorJitter')
trans_arg.add_argument('--horizon', default=0.3, help='probability of the image being flipped of RandomHorizontalFlip')
trans_arg.add_argument('--vertical', default=0.3, help='probability of the image being flipped of RandomVerticalFlip')
trans_arg.add_argument('--rotation', default=45, help='degrees of the image being rotated of RandomRotation')


### SCHEDULER ###
scheduler_arg = add_argument_group('Scheduler')
scheduler_arg.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='the name of scheduler is being used')
scheduler_arg.add_argument('--patience', type=int, default=5, help='the patience of reduce on plateou')
scheduler_arg.add_argument('--factor', type=float, default=0.5, help='the factor of reduce on plateou')
scheduler_arg.add_argument('--min_lr', type=float, default=0, help='the min lr of reduce on plateou')


### MISC ###
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--random_seed', type=int, default=1, help='random seed (default: 1)')
misc_arg.add_argument('--cuda', default=True, type=bool, help='use cuda or not')
misc_arg.add_argument('--use_cudnn', default=1, type=int, help='use cudnn or not')
misc_arg.add_argument('--uid', type=str, default=None, help='unique id for the training')
misc_arg.add_argument('--use_tensorboard', action='store_true')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--log_iter', type=int, default=200)


misc_arg.add_argument('--create_reference_images', action='store_true',
                      help='create reference frame at at desired size or not')
misc_arg.add_argument('--reference_folder', type=str, default='Reference_Images',
                      help='the name of folder contains reference images')
misc_arg.add_argument('--ref_counter', type=int, default=0,
                      help='counter reference images')

misc_arg.add_argument('--result_images_folder', type=str, default='Result_Images',
                      help='the name of folder contains result images')
misc_arg.add_argument('--out_counter', type=int, default=0,
                      help='counter result images')
misc_arg.add_argument('--graph_folder', type=str, default='graph',
                      help='the name of folder contains graph results (loss, psnr, ssim,...)')
misc_arg.add_argument('--save_path', type=str, default='', help='the output dir of weights')
misc_arg.add_argument('--test_epoch_path', type=str, default='', help='the output dir test')
misc_arg.add_argument('--timestamp', type=str, default='', help='time when execute the code')
misc_arg.add_argument('--loss_list_data', type=list, default=[], help='list to save loss value in csv file')
misc_arg.add_argument('--test_mode', type=str, choices=["test", "custom", "ssim_test"], default=False, help='test mode')
# misc_arg.add_argument('--test_custom', type=bool, default=False, help='test mode')
misc_arg.add_argument('--scale', type=int, default=4, help='scale factor of the models')
def get_args():
    """Parses all the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed



# python D:\ANH\Anh_model\models\ConvLSTM-Frame-Interpolation\main.py --datasetName 'VimeoSepTuplet'--modelName 'base3GAP' --test_batch_size 1 --checkpoint_dir "D:\ANH\Anh_model\models\ConvLSTM-Frame-Interpolation\Results\/20250316_1436_74976/\base3GAP_best.pth" --save_images True