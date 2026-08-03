import os
import torch
import torch.nn as nn 
import torch.multiprocessing.spawn
from data import train_dataloader,valid_dataloader
from utils import Adder
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F

from warmup_scheduler import GradualWarmupScheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed



import torchvision
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
        self.vgg = VGG19().cuda()
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
    state_dict = torch.load(ckpt_path)
    state_dict = state_dict['model']
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict

def _train(model, args):
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
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    best_psnr=-1
    best_ep = -1 
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        best_ep = state['Bestep']
        best_psnr = state['Best']
        model.load_state_dict(state['model'])

        print('Resume from %d'%epoch)
        epoch += 1
    elif args.test_model:
        state = torch.load(args.test_model)
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

            # label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
            # label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            # pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
            # pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            # label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
            # label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            # pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2,-1))
            # pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            # label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
            # label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            # pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2,-1))
            # pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            # f1 = criterion(pred_fft1, label_fft1)
            # f2 = criterion(pred_fft2, label_fft2)
            # f3 = criterion(pred_fft3, label_fft3)
            # loss_fft = f1+f2+f3
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
            val = _valid(model.module, args, epoch_idx, ots)
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