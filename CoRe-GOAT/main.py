from tools import run_net
from tools import test_net
from utils import parser
import torch
from utils.multi_gpu import setup_env
from mmengine.dist import (is_main_process, get_rank, init_dist,
                           is_distributed, sync_random_seed)


def main():
    # config
    print(torch.cuda.device_count())
    torch.backends.cudnn.enabled = False
    args = parser.get_args()
    if args.benchmark == 'MTL':
        if not args.usingDD:
            args.score_range = 100

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
    if args.use_bp:
        args.qk_dim = 768
    else:
        args.qk_dim = 1024

    setup_env(args.launcher, distributed=args.distributed)

    if is_main_process():
        print(args)

    # run
    if args.test:
        test_net(args)
    else:
        run_net(args)


if __name__ == '__main__':
    main()