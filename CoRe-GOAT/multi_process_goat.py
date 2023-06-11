import os
from multiprocessing import Pool

runs = 32

config_list = [[3e-5, 1024, 0],
               [1e-5, 1024, 0],
               [3e-6, 1024, 0],
               [1e-6, 1024, 0],
               [3e-7, 1024, 0],
               [1e-7, 1024, 0],
               [3e-8, 1024, 0],
               [1e-8, 1024, 0],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [1e-8, 1024, 1],
               [3e-5, 64, 0],
               [1e-5, 64, 0],
               [3e-6, 64, 0],
               [1e-6, 64, 0],
               [3e-7, 64, 0],
               [1e-7, 64, 0],
               [3e-8, 64, 0],
               [1e-8, 64, 0],
               [1e-8, 64, 1],
               [1e-8, 64, 1],
               [1e-8, 64, 1],
               [1e-8, 64, 1],
               [1e-8, 64, 1],
               [1e-8, 64, 1],
               [1e-8, 64, 1],
               [1e-8, 64, 1]]


def func(i):
    os.system(
        f'CUDA_VISIBLE_DEVICES={i % 8} bash ./scripts/train.sh MTL try --launcher=pytorch --use_goat=1 --lr={config_list[i][0]} --num_layers=4 --use_multi_gpu=1 --seed=42 --num_selected_frames=1 --random_select_frames=0 --linear_dim={config_list[i][1]} --warmup={config_list[i][2]} --max_epoch=200 --use_i3d_bb=0 --use_swin_bb=1')


pool = Pool(runs)
pool.map(func, range(runs))
pool.close()
pool.join()
