import os
from multiprocessing import Pool

runs = 16

# use_goat, use_formation, use_self, lr, warmup
use_goat_list = [1]
use_formation_list = [0]
use_self_list = [1]
lr_list = [1e-3, 3e-3, 1e-4, 3e-4, 1e-5, 3e-5, 1e-6, 3e-6]
warmup_list = [0, 1]

config_list = []
for use_goat in use_goat_list:
    if use_goat == 0:
        for lr in lr_list:
            for warmup in warmup_list:
                config_list.append([0, 0, 0, lr, warmup])
    else:
        for use_formation in use_formation_list:
            for use_self in use_self_list:
                for lr in lr_list:
                    for warmup in warmup_list:
                        if use_formation * use_self == 0:
                            config_list.append([1, use_formation, use_self, lr, warmup])


def func(i):
    os.system(
        f'CUDA_VISIBLE_DEVICES={i % 4} bash ./scripts/train.sh MTL try --launcher=none '
        f'--use_goat={config_list[i][0]} --use_formation={config_list[i][1]} --use_self={config_list[i][2]} '
        f'--lr={config_list[i][3]} --warmup={config_list[i][4]} --use_multi_gpu=1 --seed=42 '
        f'--use_i3d_bb=1 --use_swin_bb=0')


pool = Pool(runs)
pool.map(func, range(runs))
pool.close()
pool.join()
