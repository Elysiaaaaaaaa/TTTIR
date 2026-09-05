import os
import math
import argparse
import random
import logging
from models.archs.tttir_arch import build_net

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from data import train_dataloader,valid_dataloader
from utils.util import Adder
from torch.utils.tensorboard import SummaryWriter
from task_validation import eval_derain
import torch.nn.functional as F

from warmup_scheduler import GradualWarmupScheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import torchvision


import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
import cv2

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def train_enhancement(args):
    #### options
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-task', type=str, help='derain, enhance', default='./options/train/Huawei.yml')

    # parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='./options/train/Huawei.yml')
    # parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
    #                     help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    # args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  weights_only=False,
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        opt['path']['pretrain_model_G'] = opt['path']['pretrain_model_G']
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='./tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)


            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            print('train_size', train_size, 'total_iters', total_iters)
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        del resume_state
    else:
        current_step = 0
        start_epoch = 0

    best_psnr = 0.
    best_ssim = 0.
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['model'] in ['sr', 'srgan'] and rank <= 0:  # image restoration validation
                    # does not support multi-GPU validation
                    pbar = util.ProgressBar(len(val_loader))
                    avg_psnr = 0.
                    idx = 0
                    for val_data in val_loader:
                        idx += 1
                        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        sr_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                        avg_psnr += util.calculate_psnr(sr_img, gt_img)
                        pbar.update('Test {}'.format(img_name))

                    avg_psnr = avg_psnr / idx

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)
                else:  # video restoration validation
                    if opt['dist']:
                        # multi-GPU testing
                        psnr_rlt = {}  # with border and center frames
                        if rank == 0:
                            pbar = util.ProgressBar(len(val_set))

                        random_index = random.randint(0, len(val_set)-1)
                        for idx in range(rank, len(val_set), world_size):

                            if not(idx == random_index):
                                continue

                            val_data = val_set[idx]
                            val_data['LQs'].unsqueeze_(0)
                            val_data['GT'].unsqueeze_(0)
                            folder = val_data['folder']
                            idx_d, max_idx = val_data['idx'].split('/')
                            idx_d, max_idx = int(idx_d), int(max_idx)
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32,
                                                               device='cuda')
                            model.feed_data(val_data)
                            model.test()
                            visuals = model.get_current_visuals()
                            sou_img = util.tensor2img(visuals['LQ'])
                            rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                            gt_img = util.tensor2img(visuals['GT'])  # uint8
                            ill_img = util.tensor2img(visuals['ill'])
                            save_img = np.concatenate([sou_img, rlt_img, ill_img, gt_img, rlt_img3, rlt_img2], axis=0)
                            im_path = os.path.join(opt['path']['val_images'], '%06d.png' % current_step)
                            cv2.imwrite(im_path, save_img.astype(np.uint8))

                            # calculate PSNR
                            psnr_rlt[folder][idx_d] = util.calculate_psnr(rlt_img, gt_img)
                        # # collect data
                        for _, v in psnr_rlt.items():
                            dist.reduce(v, 0)
                        dist.barrier()

                        if rank == 0:
                            psnr_rlt_avg = {}
                            psnr_total_avg = 0.
                            for k, v in psnr_rlt.items():
                                psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                                psnr_total_avg += psnr_rlt_avg[k]
                            psnr_total_avg /= len(psnr_rlt)
                            log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                            for k, v in psnr_rlt_avg.items():
                                log_s += ' {}: {:.4e}'.format(k, v)
                            logger.info(log_s)
                            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                                for k, v in psnr_rlt_avg.items():
                                    tb_logger.add_scalar(k, v, current_step)
                    else:
                        # pbar = util.ProgressBar(len(val_loader))
                        psnr_rlt = {}  # with border and center frames
                        psnr_rlt_avg = {}
                        psnr_total_avg = 0.
                        ssim_rlt = {}  # with border and center frames
                        ssim_rlt_avg = {}
                        ssim_total_avg = 0.
                        for val_data in val_loader:
                            folder = val_data['folder'][0]
                            idx_d = val_data['idx']
                            # border = val_data['border'].item()
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[folder] = []
                            if ssim_rlt.get(folder, None) is None:
                                ssim_rlt[folder] = []

                            model.feed_data(val_data)
                            model.test()
                            visuals = model.get_current_visuals()
                            rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                            gt_img = util.tensor2img(visuals['GT'])  # uint8

                            # calculate PSNR
                            psnr = util.calculate_psnr(rlt_img, gt_img)
                            psnr_rlt[folder].append(psnr)
                            ssim = util.calculate_ssim(rlt_img, gt_img)
                            ssim_rlt[folder].append(ssim)
                            # pbar.update('Test {} - {}'.format(folder, idx_d))
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[k] = sum(v) / len(v)
                            psnr_total_avg += psnr_rlt_avg[k]
                        for k, v in ssim_rlt.items():
                            ssim_rlt_avg[k] = sum(v) / len(v)
                            ssim_total_avg += ssim_rlt_avg[k]
                        psnr_total_avg /= len(psnr_rlt)
                        ssim_total_avg /= len(ssim_rlt)
                        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                        log_s1 = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
                        new_best = False
                        if psnr_total_avg > best_psnr:
                            best_psnr = psnr_total_avg
                            log_s += ' best psnr : {:.4e}'.format(best_psnr)
                            new_best = True
                        if ssim_total_avg > best_ssim:
                            best_ssim = ssim_total_avg
                            log_s1 += ' best ssim : {:.4e}'.format(best_ssim)
                            new_best = True
                        logger.info(log_s)
                        logger.info(log_s1)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)

                        if new_best and current_step % opt['logger']['save_checkpoint_freq'] == 0:
                            if rank <= 0:
                                logger.info('Saving models and training states.')
                                model.save(current_step)
                                model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def init_parameters(ckpt_path):
    state_dict = torch.load(ckpt_path, weights_only=False)
    state_dict = state_dict['model']
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict

def train_derain(model, args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(42)
    accelerator.print(f'device {str(accelerator.device)} is used!')

    # sampler = DistributedSampler(train_dataloader(args.data_dir, args.batch_size, args.num_worker),shuffle=True)
    # device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')


    model, optimizer, dataloader, ots = accelerator.prepare(
        model,
        torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8),
        train_dataloader(args.data_dir, args.batch_size, args.num_worker),
        valid_dataloader(args.data_dir, batch_size=1, num_workers=args.num_worker)
    )

    model.train()

    criterion = torch.nn.L1Loss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    # dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)

    warmup_epochs=1
    # A one-epoch smoke run is valid; CosineAnnealingLR requires a positive
    # cycle length, while normal multi-epoch behaviour remains unchanged.
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.num_epoch - warmup_epochs), eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    best_psnr=-1
    best_ep = -1
    if args.resume:
        state = torch.load(args.resume, weights_only=False)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        best_ep = state['Bestep']
        best_psnr = state['Best']
        model.load_state_dict(state['model'])

        print('Resume from %d'%epoch)
        epoch += 1
    elif args.test_model:
        state = torch.load(args.test_model, weights_only=False)
        epoch = state['epoch']
        best_ep = state['Bestep']
        best_psnr = state['Best']
        state_dict = state['model']
        model.load_state_dict(state_dict)
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter() if accelerator.is_local_main_process else None
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()

    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    ssim_loss = SSIM().to(accelerator.device)
    cri_vgg = VGGLoss().to(accelerator.device)

    train_log = open(os.path.join(args.model_save_dir,'trainlog.txt'), mode = 'a',encoding='utf-8')

    for epoch_idx in range(epoch, args.num_epoch + 1):
        accelerator.wait_for_everyone()
        model.train()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data


            optimizer.zero_grad()
            pred_img = model(input_img)

            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            if len(pred_img) == 3:
                l1 = criterion(pred_img[0], label_img4)
                l2 = criterion(pred_img[1], label_img2)
                l3 = criterion(pred_img[2], label_img)
                loss_content = l1+l2+l3
                l_ssim = (1 - ssim_loss(pred_img[2], label_img))
                l_vgg = cri_vgg(pred_img[2], label_img)
            else:
                loss_content = criterion(pred_img, label_img)
                l_ssim = (1 - ssim_loss(pred_img, label_img))
                l_vgg = cri_vgg(pred_img, label_img)

            loss_fft = torch.tensor(0)



            loss = loss_content + 0.1 * loss_fft + 0.1 * l_ssim + 0.3 * l_vgg
            # loss.backward()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()



            if accelerator.is_local_main_process:

                iter_pixel_adder(loss_content.item())
                iter_fft_adder(loss_fft.item())


                epoch_pixel_adder(loss_content.item())
                epoch_fft_adder(loss_fft.item())


                if (iter_idx + 1) % args.print_freq == 0:
                    print("Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f  Loss fft: %7.4f" % (
                         epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                        iter_fft_adder.average()))
                    writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                    writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

                    iter_pixel_adder.reset()
                    iter_fft_adder.reset()



        if accelerator.is_local_main_process:
            overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
            torch.save({'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch_idx,'Best':best_psnr,'Bestep':best_ep},overwrite_name)


        if accelerator.is_local_main_process and epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch_idx,'Best':best_psnr,'Bestep':best_ep}, save_name)

            print("EPOCH: %02d\n Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
                epoch_idx, epoch_pixel_adder.average(), epoch_fft_adder.average()))
            train_log.write("EPOCH: %02d\n Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f \n" % (
                epoch_idx,  epoch_pixel_adder.average(), epoch_fft_adder.average()))
            epoch_fft_adder.reset()
            epoch_pixel_adder.reset()
            # epoch_ssim_adder.reset()

        scheduler.step()


        if accelerator.is_local_main_process and epoch_idx % args.valid_freq == 0:

            # dist.barrier()
            model.eval()
            # ``Accelerator`` only wraps the model in DDP for multi-process
            # launches.  Unwrapping handles both single-GPU and DDP training.
            val = eval_derain(accelerator.unwrap_model(model), args, epoch_idx, ots)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            print('Best PSNR %.2f at %03d epoch ' % (best_psnr,best_ep))###############
            train_log.write('%03d epoch \n Average PSNR %.2f dB\n' % (epoch_idx, val))
            train_log.write('Best PSNR %.2f at %03d epoch \n' % (best_psnr,best_ep))

            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                best_psnr = val ########
                best_ep = epoch_idx
                print('new-Best PSNR %.2f at %03d epoch ' % (best_psnr,best_ep))###############
                train_log.write('new-Best PSNR %.2f at %03d epoch \n' % (best_psnr,best_ep))
                torch.save({'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch_idx,'Best':best_psnr,'Bestep':best_ep}, os.path.join(args.model_save_dir, 'Best.pkl'))
            writer.close()


        accelerator.wait_for_everyone()

    save_name = os.path.join(args.model_save_dir, 'Final.pkl')

    torch.save({'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch_idx,'Best':best_psnr,'Bestep':best_ep},save_name)
