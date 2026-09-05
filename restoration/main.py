import os
import math
import argparse
import random
import logging
from models.archs.tttir_arch import build_net

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
import cv2
from torch.backends import cudnn

from task_train import train_derain, train_enhancement
from task_validation import eval_enhancement
from test_derain import _eval as test_derain

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['derain', 'enhance'], required=True,
                        help='Restoration task to run.')

    # enhancement
    parser.add_argument('--opt', type=str, help='Path to option YAML file.', default='./options/train/Huawei.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    # derain
    # Directories
    parser.add_argument('--model_name', default='IRderain',type=str)
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=tuple, default=(256,256))
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--num_res', type=int, default=8)

    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    if args.task == 'enhance':
        if args.mode == 'train':
            train_enhancement(args)
        elif args.mode == 'test':
            eval_enhancement(args)
    else:
        args.model_save_dir = os.path.join('results/',args.model_name)
        args.result_dir = os.path.join('results/', args.model_name, 'test')
        # args.result_dir = '/data/goodata/IRNeXt/Dehazing/OTS/rhaze/results/'
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        command = 'cp ' + 'models/archs/tttir_layers.py ' + args.model_save_dir
        os.system(command)
        command = 'cp ' + 'models/archs/tttir_arch.py ' + args.model_save_dir
        os.system(command)
        command = 'cp ' + 'task_train.py ' + args.model_save_dir
        os.system(command)
        command = 'cp ' + 'main.py ' + args.model_save_dir
        os.system(command)
        print(args)
        cudnn.benchmark = True
        if not os.path.exists('results/'):
            os.makedirs(args.model_save_dir)
        # if not os.path.exists('results/' + args.model_name + '/'):
        #     os.makedirs('results/' + args.model_name + '/')
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        model = build_net()
        print(model)
        if args.mode == 'train':
            train_derain(model, args)

        elif args.mode == 'test':
            test_derain(model, args)

if __name__ == '__main__':
    main()
