import os
from multiprocessing import Pool

runs = 4

# use_goat, use_formation, use_self, lr, warmup
use_goat_list = [1]
use_formation_list = [1]
use_self_list = [0]
lr_list = [7e-6, 1e-5, 3e-4, 3e-5]
warmup_list = [0]

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
                        if use_formation * use_self == 0 and use_formation + use_self != 0:
                            config_list.append([1, use_formation, use_self, lr, warmup])


def func(i):
    os.system(
        f'CUDA_VISIBLE_DEVICES={i % 4} python3 -u main.py --launcher=none --type USDL --use_goat={config_list[i][0]} '
        f'--use_formation={config_list[i][1]} --use_self={config_list[i][2]} --lr={config_list[i][3]} '
        f'--use_multi_gpu=1 --seed=42 --num_selected_frames=1 --random_select_frames=0 --linear_dim=1024 '
        f'--warmup={config_list[i][4]} --use_i3d_bb=0 --use_swin_bb=1')


pool = Pool(runs)
pool.map(func, range(runs))
pool.close()
pool.join()
