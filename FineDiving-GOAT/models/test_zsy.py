import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
from models.Backbone import *
import torch.nn as nn
import time
import numpy as np
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d
from PIL import Image
import glob
from datasets.FineDiving_Pair import FineDiving_Pair_Dataset
from torchvideotransforms import video_transforms, volume_transforms

train_trans = video_transforms.Compose([
    video_transforms.RandomHorizontalFlip(),
    video_transforms.Resize((200, 112)),
    video_transforms.RandomCrop(112),
    volume_transforms.ClipToTensor(),
    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

start_idx = list(range(0, 90, 10))
a = torch.ones((3, 96, 5, 6))
a = torch.cat([a[:, i: i + 16] for i in start_idx])
print(a.size())
image_list = sorted((glob.glob(os.path.join('/home/shiyi_ys/AQA/FineDiving-main/FineDiving-main/datasets/FINADiving_MTL_256s', '01', '1', '*.jpg'))))
video = [Image.open(image_list[i]) for i in range(len(image_list))]
video = train_trans(video).float().cuda()
print(video.size())
total_video = torch.cat((video, video), 0)
print(total_video.size())
video_pack = torch.cat([total_video[:, :, i: i + 16] for i in start_idx])
print(video_pack.size())
base_model = I3D_backbone(I3D_class=400)
base_model.load_pretrain('models/model_rgb.pth')
video = torch.ones((8, 3, 96, 112, 112))
com_feature_12, com_feamap_12 = base_model(video, video)
print(com_feature_12.size())
print(com_feamap_12.size())
