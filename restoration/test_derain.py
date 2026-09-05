import os
import torch
from torchvision.transforms import functional as F
from utils.util import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim
import torch.nn.functional as f

from skimage import img_as_ubyte

# ---------------------------------------------------

def _eval(model, args):
    # Checkpoints are supplied explicitly by the local user.  PyTorch >= 2.6
    # otherwise defaults to ``weights_only=True``, which cannot read legacy
    # optimizer metadata stored in the project's checkpoints.
    state_dict = torch.load(args.test_model, weights_only=False)
    state_dict = state_dict['model']
    # for k,v in state_dict.items():
    #     print(k)
    # for name, param in model.named_parameters():
    #     print('Parameter name:', name)
    # print(model)
    
    # for k,v in state_dict.items():
    #     print(k)

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(state_dict)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # model = model.to(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    factor = 32
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, _, label_img, name = data
            input_img = input_img.to(device)
            # input_img_yuv = input_img_yuv.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            # print('h,w',h,w)
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            # print('H,W',H,W)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
            # input_img_yuv = f.pad(input_img_yuv, (0, padw, 0, padh), 'reflect')
            # print('input',input_img.shape)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tm = time.time()

            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]
            if device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            label_img = label_img.to(device)
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))	
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)	
            #print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                # save_gt_name = os.path.join('/data/goodata/RestoreNet/results2/gopro1/gt', name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

                # label_img += 0.5 / 255
                # label_img = F.to_pil_image(label_img.squeeze(0).cpu(), 'RGB')
                # label_img.save(save_gt_name)

            psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr_val)

            #print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.4f dB' % (ssim_adder.average()))

        print("Average time: %f" % adder.average())
