import random
import numpy as np
import logging
from typing import Union, Optional
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel.distributed import DistributedDataParallel
from mmengine.runner import set_random_seed
from mmengine.device import get_device
from mmengine.dataset import DefaultSampler
from mmengine.utils.dl_utils import set_multi_processing
from mmengine.model import is_model_wrapper
from mmengine.dist import (is_main_process, get_rank, init_dist,
                           is_distributed, sync_random_seed)


def setup_env(
        launcher: str,
        distributed: bool,
        cudnn_benchmark: bool = False,
        backend: str = 'nccl') -> None:
    if cudnn_benchmark:
        # Whether to use `cudnn.benchmark` to accelerate training.
        torch.backends.cudnn.benchmark = True
    set_multi_processing(distributed=distributed)

    if distributed and not is_distributed():
        init_dist(launcher, backend=backend)


def wrap_model(model: nn.Module,
               distributed: bool) -> Union[DistributedDataParallel, nn.Module]:
    # Set `export CUDA_VISIBLE_DEVICES=-1` to enable CPU training.
    model = model.to(get_device())

    if not distributed:
        return model

    model = DistributedDataParallel(
        module=model,
        device_ids=[int(os.environ['LOCAL_RANK'])],
        broadcast_buffers=False,
        find_unused_parameters=False)
    return model


def build_dataloader(
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        persistent_workers: bool = True,
        seed: Optional[int] = None
) -> DataLoader:
    sampler = DefaultSampler(dataset, shuffle=shuffle, seed=seed)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers)
    return dataloader


def set_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = sync_random_seed()
    else:
        set_random_seed(seed=seed)

    print(f"Set seed as: {seed}")

    # if get_rank() == 0:
    #     # Only master rank will print msg
    #     print(f"Set seed as: {seed}")

    return seed


def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


def worker_init_fn(worker_id):
    """
    Init worker in dataloader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def init_seed(args):
    """
    Set random seed for torch and numpy.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args:
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B = X.shape[0]

    rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
    ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

    dist = rx - 2.0 * X.matmul(Y.transpose(1, 2)) + ry.transpose(1, 2)

    return torch.sqrt(dist)


def log_best(rho_best, RL2_best, epoch_best, args):
    # log for best
    with open(args.result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if args.use_goat:
            if args.use_formation:
                mode = 'Formation'
            elif args.use_bp:
                mode = 'BP'
            elif args.use_self:
                mode = 'SELF'
            else:
                mode = 'GOAT'
        else:
            mode = 'Ori'

        if args.use_i3d_bb:
            backbone = 'I3D'
        elif args.use_swin_bb:
            backbone = 'SWIN'
        else:
            backbone = 'BP_BB'

        log_list = [format(rho_best, '.4f'), epoch_best, args.use_goat, args.lr, args.num_epochs, args.warmup,
                    args.seed, args.train_backbone, args.num_selected_frames, args.num_heads,
                    args.num_layers, args.random_select_frames, args.train_batch_size, args.test_batch_size, args.linear_dim, RL2_best, mode, backbone]
        writer.writerow(log_list)
