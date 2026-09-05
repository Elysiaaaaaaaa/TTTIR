import cv2
import os.path as osp
import logging
import argparse
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

def eval_enhancement(args):
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    save_imgs = True
    model = create_model(opt)
    # Keep enhancement outputs under the same configured results root as the
    # other restoration tasks instead of creating a hard-coded dataset folder.
    save_folder = opt['path']['results_root']
    util.mkdirs(save_folder)

    print('mkdir finish')
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)

    for phase, dataset_opt in opt['datasets'].items():
        # 为每个数据集创建独立的子目录
        dataset_save_folder = osp.join(save_folder, phase)  # 使用 phase (Rain100L, Rain100H)

        GT_folder = osp.join(dataset_save_folder, 'images/GT')
        output_folder = osp.join(dataset_save_folder, 'images/output')
        input_folder = osp.join(dataset_save_folder, 'images/input')

        util.mkdirs(GT_folder)
        util.mkdirs(output_folder)
        util.mkdirs(input_folder)

        print(f'Processing dataset: {phase}')
        print(f'Saving to: {dataset_save_folder}')

        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        for val_data in val_loader:
            model.feed_data(val_data)
            model.test()
            visuals = model.get_current_visuals()
            rlt_img = util.tensor2img(visuals['rlt'])
            gt_img = util.tensor2img(visuals['GT'])
            input_img = util.tensor2img(visuals['LQ'])

            if save_imgs:
                try:
                    # 获取原始文件名（不包含路径）
                    # 假设 val_data['folder'] 或 val_data['LQ_path'] 包含文件名信息
                    if 'LQ_path' in val_data:
                        filename = osp.basename(val_data['LQ_path'])
                    elif 'folder' in val_data:
                        # 原来的提取方式，但只取文件名
                        tag = str(val_data['folder']).split('[')[1].split(']')[0]
                        tag = tag.replace("'", "")
                        filename = tag
                    else:
                        # 使用索引作为文件名
                        filename = f"image_{idx}.png"

                    # 保存到对应的数据集目录
                    cv2.imwrite(osp.join(output_folder, filename), rlt_img)
                    cv2.imwrite(osp.join(GT_folder, filename), gt_img)
                    cv2.imwrite(osp.join(input_folder, filename), input_img)

                    print(f'Saved: {phase}/{filename}')

                except Exception as e:
                    print(f'Error saving {phase}: {e}')
                    import ipdb; ipdb.set_trace()



import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils.util import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f
from derain_metrics import *


def eval_derain(model, args, ep, ots):

    # ots = valid_dataloader(args.data_dir, batch_size=1, num_workers=2)
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 32
        for idx, data in enumerate(ots):
            input_img, label_img = data

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
            #new
            # image_yuv = f.pad(image_yuv, (0, padw, 0, padh), 'reflect')


            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)
            if len(pred) == 3:
                pred = pred[2]

            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            # psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            psnr = calculate_psnr(p_numpy * 255, label_numpy * 255, crop_border=0, test_y_channel=True)

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()
