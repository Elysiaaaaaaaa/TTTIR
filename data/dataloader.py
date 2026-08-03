import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_dataloader(path, batch_size=64, num_workers=0):
    # image_dir = os.path.join(path, 'train/Rain13K')
########
    image_dir = path
########
    dataloader = DataLoader(
        #DeblurDataset(image_dir, ps=256),
        DeblurDataset_siting(image_dir,ps=256),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    # image_dir = os.path.join(path, 'test/Rain100H')
    ########
    image_dir = path
########
    dataloader = DataLoader(
        #DeblurDataset(image_dir, is_test=True),
        DeblurDataset_siting(image_dir,is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    ########
    image_dir = path
########
    dataloader = DataLoader(
        #DeblurDataset(os.path.join(path, 'test'), is_valid=True),
        DeblurDataset_siting(image_dir,is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

import random


class DeblurDataset_siting(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):

        if is_test or is_valid:
            # self.image_dir = os.path.join(image_dir,'test/Rain100H')
            self.image_dir = os.path.join(image_dir,'test/Rain100L')
            # self.image_dir = os.path.join(image_dir,'test/Test1200')
            # self.image_dir = os.path.join(image_dir,'Test')
            self.image_list = os.listdir(os.path.join(self.image_dir,'input'))
            
        else:
            self.image_dir = os.path.join(image_dir,'train')
            self.image_list = os.listdir(os.path.join(self.image_dir,'input'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input',self.image_list[idx])).convert('RGB')
        image_yuv = image.convert('YCbCr')
        if self.is_valid or self.is_test:      
            label = Image.open(os.path.join(self.image_dir,'target', self.image_list[idx].replace('rain','clean'))).convert('RGB')
        else:
            label = Image.open(os.path.join(self.image_dir,'target', self.image_list[idx].replace('rain','clean'))).convert('RGB')
        ps = self.ps

        if self.ps is not None:
            width,height = image.size
            if width < self.ps or height < self.ps:
                width = max(width,260)
                height = max(height,260)
                image = image.resize((width, height), Image.BILINEAR)
                image_yuv = image_yuv.resize((width, height), Image.BILINEAR)
                label = label.resize((width, height), Image.BILINEAR)

            image = F.to_tensor(image)
            image_yuv = F.to_tensor(image_yuv)
            label = F.to_tensor(label)

            hh, ww = label.shape[1], label.shape[2]

            rr = random.randint(0, hh-ps)
            cc = random.randint(0, ww-ps)
            # rr = cc = 0
            
            image = image[:, rr:rr+ps, cc:cc+ps]
            image_yuv = image_yuv[:, rr:rr+ps, cc:cc+ps]
            label = label[:, rr:rr+ps, cc:cc+ps]

            if random.random() < 0.5:
                image = image.flip(2)
                image_yuv = image_yuv.flip(2)
                label = label.flip(2)
        else:
            image = F.to_tensor(image)
            image_yuv = F.to_tensor(image_yuv)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx]
            return image, image_yuv, label, name
        # return image, image_yuv, label
        return image, label


    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'PNG']:
                raise ValueError



























# import os
# import torch
# import numpy as np
# from PIL import Image as Image
# from torchvision.transforms import functional as F
# from torch.utils.data import Dataset, DataLoader
# from PIL import ImageFile
# import cv2
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# def train_dataloader(path, batch_size=64, num_workers=0):
#     # image_dir = os.path.join(path, 'train/Rain13K')
# ########
#     image_dir = path
# ########
#     dataloader = DataLoader(
#         #DeblurDataset(image_dir, ps=256),
#         DeblurDataset_siting(image_dir,ps=256),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#     return dataloader


# def test_dataloader(path, batch_size=1, num_workers=0):
#     # image_dir = os.path.join(path, 'test/Rain100H')
#     ########
#     image_dir = path
# ########
#     dataloader = DataLoader(
#         #DeblurDataset(image_dir, is_test=True),
#         DeblurDataset_siting(image_dir,is_test=True),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )

#     return dataloader


# def valid_dataloader(path, batch_size=1, num_workers=0):
#     ########
#     image_dir = path
# ########
#     dataloader = DataLoader(
#         #DeblurDataset(os.path.join(path, 'test'), is_valid=True),
#         DeblurDataset_siting(image_dir,is_valid=True),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers
#     )

#     return dataloader

# import random


# class DeblurDataset_siting(Dataset):
#     def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):

#         if is_test or is_valid:
#             self.image_dir = os.path.join(image_dir,'test/Rain100L')
#             self.image_list = os.listdir(os.path.join(self.image_dir,'input'))
            
#         else:
#             self.image_dir = os.path.join(image_dir,'train')
#             self.image_list = os.listdir(os.path.join(self.image_dir,'input'))
#         self._check_image(self.image_list)
#         self.image_list.sort()
#         self.transform = transform
#         self.is_test = is_test
#         self.is_valid = is_valid
#         self.ps = ps
    
#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         image = Image.open(os.path.join(self.image_dir, 'input',self.image_list[idx])).convert('RGB')
#         image_yuv = image.convert('YCbCr')
#         if self.is_valid or self.is_test:      
#             label = Image.open(os.path.join(self.image_dir,'target', self.image_list[idx].replace('rain','clean'))).convert('RGB')
#         else:
#             label = Image.open(os.path.join(self.image_dir,'target', self.image_list[idx].replace('rain','clean'))).convert('RGB')
#         ps = self.ps

#         if self.ps is not None:
#             width,height = image.size
#             if width < self.ps or height < self.ps:
#                 width = max(width,260)
#                 height = max(height,260)
#                 image = image.resize((width, height), Image.BILINEAR)
#                 image_yuv = image_yuv.resize((width, height), Image.BILINEAR)
#                 label = label.resize((width, height), Image.BILINEAR)

#             image = F.to_tensor(image)
#             image_yuv = F.to_tensor(image_yuv)
#             label = F.to_tensor(label)

#             hh, ww = label.shape[1], label.shape[2]

#             rr = random.randint(0, hh-ps)
#             cc = random.randint(0, ww-ps)
#             # rr = cc = 0
            
#             image = image[:, rr:rr+ps, cc:cc+ps]
#             image_yuv = image_yuv[:, rr:rr+ps, cc:cc+ps]
#             label = label[:, rr:rr+ps, cc:cc+ps]

#             if random.random() < 0.5:
#                 image = image.flip(2)
#                 image_yuv = image_yuv.flip(2)
#                 label = label.flip(2)
#         else:
#             image = F.to_tensor(image)
#             image_yuv = F.to_tensor(image_yuv)
#             label = F.to_tensor(label)

#         if self.is_test:
#             name = self.image_list[idx]
#             return image, image_yuv, label, name
#         # return image, image_yuv, label
#         return image, label


#     @staticmethod
#     def _check_image(lst):
#         for x in lst:
#             splits = x.split('.')
#             if splits[-1] not in ['png', 'jpg', 'jpeg', 'PNG']:
#                 raise ValueError


