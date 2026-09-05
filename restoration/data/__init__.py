"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data
from .data_aug import PairRandomCrop, PairCompose, PairRandomHorizontalFilp, PairToTensor
from .derain_dataloader import train_dataloader, test_dataloader, valid_dataloader


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            # Device visibility is controlled outside the YAML.  Preserve the
            # per-GPU worker convention, with one worker group on CPU.
            visible_gpus = max(1, torch.cuda.device_count())
            num_workers = dataset_opt['n_workers'] * visible_gpus
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        # CPU-only environments may not permit multiprocessing IPC.  GPU
        # evaluation keeps the original single worker; CPU evaluation remains
        # functionally identical with synchronous loading.
        num_workers = 1 if torch.cuda.is_available() else 0
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                           pin_memory=False)


def create_dataset(dataset_opt):

    from data.lowlight_paired_dataset import ll_dataset as D
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
