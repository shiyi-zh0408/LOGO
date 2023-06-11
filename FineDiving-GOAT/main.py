import os
import sys

import torch
from tools import train_net, test_net
from utils.parser import get_args
from utils.goat_utils import setup_env, init_seed
from mmengine.dist import is_main_process


def main():
    print(torch.cuda.device_count())
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    args = get_args()
    if args.use_bp:
        args.qk_dim = 768
    else:
        args.qk_dim = 1024
    args.benchmark = 'FineDiving'
    if is_main_process():
        print(args)

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True

    setup_env(args.launcher, distributed=args.distributed)
    init_seed(args)

    if args.test:
        test_net(args)
    else:
        train_net(args)


if __name__ == '__main__':
    main()
