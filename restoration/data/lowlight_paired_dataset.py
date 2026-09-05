import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os
import functools

class ll_dataset(data.Dataset):
    def __init__(self, opt):
        super(ll_dataset, self).__init__()
        self.opt = opt
        self.GT_root, self.LQ_root, self.NEG_root  = opt['dataroot_GT'], opt['dataroot_LQ'], opt['dataroot_NEG']
        self.lightmap_root = opt['lightmap_GT']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'path_NEG': [],'degradation': [], 'folder': [], 'idx': [], 'border': []}
        # Generate data info and cache data
        self.invention_num = opt['invention_num']
        self.imgs_LQ, self.imgs_GT, self.imgs_NEG = {}, {}, {}

        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        self.all_GT_paths = subfolders_GT.copy()

        count = 0
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_LQ)
            img_paths_LQ = [subfolder_LQ]
            img_paths_GT = [subfolder_GT]
            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            self.data_info['idx'].append('{}/{}'.format(count, len(subfolder_LQ)))
            self.imgs_LQ[subfolder_name] = img_paths_LQ
            self.imgs_GT[subfolder_name] = img_paths_GT

            count += 1

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        img_LQ_path = self.imgs_LQ[folder][0]
        img_GT_path = self.imgs_GT[folder][0]
        img_LQ_path = [img_LQ_path]
        img_GT_path = [img_GT_path]

        img_LQ = util.read_img_seq_pil(img_LQ_path)
        img_GT = util.read_img_seq_pil(img_GT_path)

        if self.opt['phase'] == 'train':
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

            GT_size = self.opt['GT_size']
            _, H, W = img_GT.shape
            
            
            
            if H < GT_size or W < GT_size:
                # 放大到至少 GT_size + 4 的余量
                scale_h = max(1.0, (GT_size + 4) / H)
                scale_w = max(1.0, (GT_size + 4) / W)
                new_H = int(H * scale_h)
                new_W = int(W * scale_w)
                
                # 同时放大 LQ 和 GT
                img_GT = F.interpolate(img_GT.unsqueeze(0), size=(new_H, new_W), mode='bilinear').squeeze(0)
                img_LQ = F.interpolate(img_LQ.unsqueeze(0), size=(new_H, new_W), mode='bilinear').squeeze(0)
            
            # ✅ 确保 LQ 和 GT 尺寸一致
            if img_LQ.shape[1:] != img_GT.shape[1:]:
                img_GT = F.interpolate(img_GT.unsqueeze(0), size=img_LQ.shape[1:], mode='bilinear').squeeze(0)
                
            
            
            
            _, H, W = img_GT.shape
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_LQ = img_LQ[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            rlt = util.augment_torch(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]


            return {
                'LQs': img_LQ,
                'GT': img_GT,
                'folder': folder,
                'idx': self.data_info['idx'][index],
                'border': 0
            }

        else:
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

            return {
                'LQs': img_LQ,
                'GT': img_GT,
                'folder': folder,
                'idx': self.data_info['idx'][index],
                'border': 0
            }
    def __len__(self):
        return len(self.data_info['path_LQ'])
