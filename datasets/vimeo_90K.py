import os.path
import random
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# class Vimeo_90K(Dataset):
#     def __init__(self, root, is_training):
#         self.root = root
#         self.image_root = os.path.join(self.root, 'sequences')
#         self.is_training = is_training
#
#         self.load_data()
#
#         if self.is_training:
#             self.transforms = transforms.Compose([
#                 # transforms.RandomCrop(256),
#                 transforms.Resize(size=(256, 256)),
#                 # transforms.RandomHorizontalFlip(0.5),
#                 # transforms.RandomVerticalFlip(0.5),
#                 # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
#                 transforms.ToTensor(),
#
#             ])
#         else:
#             self.transforms = transforms.Compose([
#                 transforms.Resize(size=(256, 256)),
#                 transforms.ToTensor(),
#             ])
#
#     def load_data(self):
#         self.foldersPath = []
#         self.framesPath = []
#
#         if self.is_training:
#             data_fn = os.path.join(self.root, 'tri_trainlist4.txt')
#             with open(data_fn, 'r') as f:
#                 self.datalist = f.read().splitlines()
#         else:
#             data_fn = os.path.join(self.root, 'tri_testlist2.txt')
#             with open(data_fn, 'r') as f:
#                 self.datalist = f.read().splitlines()
#
#         for item in self.datalist:
#             name = str(item).strip()
#             if len(name) <= 1:
#                 continue
#
#             framePath = sorted(glob.glob(os.path.join(self.image_root, name, '*.png')))
#
#             self.foldersPath.append(name)
#             self.framesPath.append(framePath)
#
#
#
#     def __getitem__(self, index):
#         # if self.is_training:
#         #     imgpath = os.path.join(self.image_root, self.trainlist[index])
#         # else:
#         #     imgpath = os.path.join(self.image_root, self.testlist[index])
#         # imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
#
#         images = [Image.open(self.framesPath[index][i]) for i in range(len(self.framesPath[index]))]
#
#         # Data augmentation
#         if self.is_training:
#             seed = random.randint(0, 2 ** 32)
#             images_ = []
#             for img_ in images:
#                 random.seed(seed)
#                 images_.append(self.transforms(img_))
#             images = images_
#
#             if random.uniform(0, 1) < 0.5:      # Random Temporal Flip
#                 images = images[::-1]
#             if random.uniform(0, 1) < 0.5:      # Horizontal Flip
#                 images = [TF.hflip(images[i]) for i in range(len(images))]
#             if random.uniform(0, 1) < 0.5:      # Vertical Flip
#                 images = [TF.vflip(images[i]) for i in range(len(images))]
#
#             gt = images[len(images) // 2]
#             images = images[:len(images) // 2] + images[len(images) // 2 + 1:]
#         else:
#             images = [self.transforms(img_) for img_ in images]
#
#             gt = images[len(images) // 2]
#             images = images[:len(images) // 2] + images[len(images) // 2 + 1:]
#
#         return images, gt, self.foldersPath[index]
#
#     def __len__(self):
#         return len(self.datalist)

class VimeoSepTuplet(Dataset):
    def __init__(self, root, is_training, input_frames="1357", mode='full'):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        Outputs
            frames: list of 4 frames
            gt : grouth truth frames
        """
        self.root = root
        self.image_root = os.path.join(self.root, 'sequences')
        self.is_training = is_training
        self.inputs = input_frames
        self.edge_extractor = SobelEdgeExtractor() #edge

        train_fn = os.path.join(self.root, 'sep_trainlist1.txt')
        test_fn = os.path.join(self.root, 'sep_testlist1.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        # reduce the number of test images if mode == 'mini'
        if mode != 'full':
            tmp = []
            for i, value in enumerate(self.testlist):
                if i % 38 == 0:
                    tmp.append(value)
            self.testlist = tmp

        if self.is_training:
            self.transforms = transforms.Compose([
                # transforms.RandomCrop(256),
                transforms.Resize(size=(256, 256)),
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomVerticalFlip(0.5),
                # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor(),

            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        if self.is_training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])

        imgpaths = [imgpath + f'/im{i}.png' for i in range(1, 8)]

        # Load images
        images = [Image.open(img) for img in imgpaths]

        # Select inputs and ground truth frames
        input_indices = [0, 2, 4, 6]  # Corresponding to 0th, 2nd, 4th, 6th frames
        gt_indices = [1, 3, 5]  # Ground truth frames between input frames

        input_images = [images[i] for i in input_indices]
        gt_images = [images[i] for i in gt_indices]


        # Data augmentation for training
        if self.is_training:
            seed = random.randint(0, 2 ** 32)  # Seed for reproducible augmentations
            input_images_aug = []
            gt_images_aug = []

            # Apply same augmentation to all input and GT frames
            for img in input_images:
                random.seed(seed)
                input_images_aug.append(self.transforms(img))

            for img in gt_images:
                random.seed(seed)
                gt_images_aug.append(self.transforms(img))

            input_images = input_images_aug
            gt_images = gt_images_aug

            # Random temporal flip
            if random.random() >= 0.5:
                input_images = input_images[::-1]
                gt_images = gt_images[::-1]
            # Stack for batch processing
            input_tensor = torch.stack(input_images)  # (T, C, H, W)
            edge_tensor = self.edge_extractor(input_tensor)  # (T, C, H, W)

            # Re-split
            edge_inputs = [edge_tensor[i] for i in range(4)]
            return input_images, gt_images, edge_inputs
        else:
            # Apply transforms for testing
            input_images = [self.transforms(img) for img in input_images]
            gt_images = [self.transforms(img) for img in gt_images]
            # print(f"Input images shape: {[img.shape for img in input_images]}")
            # print(f"GT images shape: {[img.shape for img in gt_images]}")

            # Stack for batch processing
            input_tensor = torch.stack(input_images)  # (T, C, H, W)
            edge_tensor = self.edge_extractor(input_tensor)  # (T, C, H, W)

            # Re-split
            edge_inputs = [edge_tensor[i] for i in range(4)]
            return input_images, gt_images, edge_inputs, imgpath

    def __len__(self):
        if self.is_training:
            return len(self.trainlist)
        else:
            return len(self.testlist)
class SobelEdgeExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, images):
        gray = images.mean(dim=1, keepdim=True)
        edge_x = torch.nn.functional.conv2d(gray, self.sobel_x, padding=1)
        edge_y = torch.nn.functional.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edge_map = magnitude / (magnitude.amax(dim=[2, 3], keepdim=True) + 1e-6)
        return edge_map.expand_as(images)
# def mean_std_calculation(dataset_path):
#     mean, std = 0.0, 0.0
#     # mean1, std1 = 0.0, 0.0
#     total_image_count = 0
#     dataset = Vimeo_90K(root=dataset_path, is_training=False)
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
#     for imgs, _ in tqdm(dataloader):
#         # input = torch.cat(imgs, dim=1)
#         inp1, inp2 = imgs[0], imgs[1]
#         batch_samples = inp1.size(0)
#         inp1 = inp1.view(batch_samples, inp1.size(1), -1)
#         # inp2 = inp2.view(batch_samples, inp2.size(1), -1)
#         mean += inp1.mean(2).sum(0)
#         std += inp1.std(2).sum(0)
#         total_image_count += batch_samples
#
#     mean /= total_image_count
#     std /= total_image_count
#
#     print(total_image_count)
#     print(mean, std)


def get_loader(mode, data_root, batch_size, shuffle, num_workers):
    if mode == 'train':
        is_training =True
    else:
        is_training = False
    dataset = VimeoSepTuplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)

if __name__ == '__main__':
    dataset = 'D:\\KIEN\\Dataset\\Vimeo_90K\\'
    # mean_std_calculation(dataset)
