import torch
import numpy as np
import os
import pickle
import random
import glob
from os.path import join
from PIL import Image


def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


data_root = '/home/shiyi_ys/AQA/FineDiving-main/FineDiving-main/datasets/FINADiving_MTL_256s'
label_path = '/home/shiyi_ys/AQA/FineDiving-main/FineDiving-main/Annotations/fine-grained_annotation_aqa.pkl'
train_split = '/home/shiyi_ys/AQA/FineDiving-main/FineDiving-main/Annotations/train_split.pkl'
test_split = '/home/shiyi_ys/AQA/FineDiving-main/FineDiving-main/Annotations/test_split.pkl'
fine_anno = '/home/shiyi_ys/AQA/FineDiving-main/FineDiving-main/Annotations/FineDiving_fine-grained_annotation.pkl'
coarse_anno = '/home/shiyi_ys/AQA/FineDiving-main/FineDiving-main/Annotations/FineDiving_coarse_annotation.pkl'
data_anno = read_pickle(label_path)
with open(train_split, 'rb') as f:
    train_dataset_list = pickle.load(f)
with open(test_split, 'rb') as f:
    test_dataset_list = pickle.load(f)
with open(fine_anno, 'rb') as f:
    fine_anno_list = pickle.load(f)
with open(coarse_anno, 'rb') as f:
    coarse_anno_list = pickle.load(f)
for i in data_anno:
    image_list = sorted((glob.glob(os.path.join(data_root, i[0], str(i[1]), '*.jpg'))))
    start_frame = int(image_list[0].split("/")[-1][:-4])
    end_frame = int(image_list[-1].split("/")[-1][:-4])
    expect_len_frame = end_frame - start_frame + 1
    true_len_frame = len(image_list)
    if true_len_frame != expect_len_frame:
        for j in range(len(image_list) - 1):
            if int(image_list[j].split("/")[-1][:-4]) + 1 != int(image_list[j + 1].split("/")[-1][:-4]):
                print(str(i), ':', str(int(image_list[j].split("/")[-1][:-4]) + 1), '\n')

    coarse = coarse_anno_list[i]
    # if len(data_anno.get(i)[4]) < len(image_frame_idx):
    #     print(i)
    if coarse['start_frame'] != start_frame:
        print(i, ' start frame in coarse annotation:', coarse['start_frame'], '; start frame in dataset:',
              start_frame)
    if coarse['end_frame'] != end_frame:
        print(i, ' end frame in coarse annotation:', coarse['end_frame'], '; end frame in dataset:',
              end_frame)
    # if len(image_list) >= 96:
    #     frame_list_new = np.linspace(start_frame, end_frame, 96).astype(np.int)
    #     image_frame_idx = [frame_list_new[j] - start_frame for j in range(96)]
    #
    #     video = [Image.open(image_list[image_frame_idx[j]]) for j in range(96)]
    #     coarse = coarse_anno_list[i]
    #     # if len(data_anno.get(i)[4]) < len(image_frame_idx):
    #     #     print(i)
    #     if coarse['start_frame'] != start_frame:
    #         print(i, ' start frame in coarse annotation:', coarse['start_frame'], '; start frame in dataset:',
    #               start_frame)
    #     if coarse['end_frame'] != end_frame:
    #         print(i, ' end frame in coarse annotation:', coarse['end_frame'], '; end frame in dataset:',
    #               end_frame)
    #     if len(data_anno.get(i)[4]) != len(image_list):
    #         print(i, '标签长度和视频长度不同')

