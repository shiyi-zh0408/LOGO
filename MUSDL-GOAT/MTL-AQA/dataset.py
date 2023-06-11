# Adapted from the code for paper 'What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment'.
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import pickle as pkl
from opts import *
from scipy import stats
from torchvideotransforms import video_transforms, volume_transforms


class VideoDataset(Dataset):

    def __init__(self, mode, args):
        super(VideoDataset, self).__init__()

        # train or test
        self.mode = mode

        # loading annotations, I3D features, CNN features, boxes annotations, formation features and bp features
        if args.use_i3d_bb:
            args.feature_path = args.i3d_feature_path
        elif args.use_swin_bb:
            args.feature_path = args.swin_feature_path
        else:
            args.feature_path = args.bpbb_feature_path
        self.args = args
        self.annotations = pkl.load(open(args.anno_path, 'rb'))
        self.keys = pkl.load(open(f'/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/{self.mode}_split{args.split}.pkl', 'rb'))
        self.feature_dict = pkl.load(open(args.feature_path, 'rb'))
        self.boxes_dict = pkl.load(open(args.boxes_path, 'rb'))
        self.cnn_feature_dict = pkl.load(open(args.cnn_feature_path, 'rb'))
        self.formation_features_dict = pkl.load(open(args.formation_feature_path, 'rb'))
        self.bp_feature_path = args.bp_feature_path
        print(f'len of {self.mode}:', len(self.keys))

        # parameters of videos
        self.data_path = args.data_path
        self.length = args.length
        self.img_size = args.img_size
        self.num_boxes = args.num_boxes
        self.out_size = args.out_size
        self.num_selected_frames = args.num_selected_frames

        # transforms
        self.transforms = video_transforms.Compose([
            video_transforms.Resize(self.img_size),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def proc_label(self, data):
        # Scores of MTL dataset ranges from 0 to 104.5, we normalize it into 0~100
        tmp = stats.norm.pdf(np.arange(output_dim['USDL']),
                             loc=data['final_score'] * (output_dim['USDL'] - 1) / label_max,
                             scale=self.args.std).astype(
            np.float32)
        data['soft_label'] = tmp / tmp.sum()

    def load_video(self, frames_path):
        length = self.length
        transforms = self.transforms
        image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
        if len(image_list) >= length:
            start_frame = int(image_list[0].split("/")[-1][:-4])
            end_frame = int(image_list[-1].split("/")[-1][:-4])
            frame_list = np.linspace(start_frame, end_frame, length).astype(np.int)
            image_frame_idx = [frame_list[i] - start_frame for i in range(length)]
            video = [Image.open(image_list[image_frame_idx[i]]) for i in range(length)]
            return transforms(video).transpose(0, 1), image_frame_idx
        else:
            T = len(image_list)
            img_idx_list = np.arange(T)
            img_idx_list = img_idx_list.repeat(2)
            idx_list = np.linspace(0, T * 2 - 1, length).astype(np.int)
            image_frame_idx = [img_idx_list[idx_list[i]] for i in range(length)]

            video = [Image.open(image_list[image_frame_idx[i]]) for i in range(length)]
            return transforms(video).transpose(0, 1), image_frame_idx

    def load_idx(self, frames_path):
        length = self.length
        image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
        if len(image_list) >= length:
            start_frame = int(image_list[0].split("/")[-1][:-4])
            end_frame = int(image_list[-1].split("/")[-1][:-4])
            frame_list = np.linspace(start_frame, end_frame, length).astype(np.int)
            image_frame_idx = [frame_list[i] - start_frame for i in range(length)]
            return image_frame_idx
        else:
            T = len(image_list)
            img_idx_list = np.arange(T)
            img_idx_list = img_idx_list.repeat(2)
            idx_list = np.linspace(0, T * 2 - 1, length).astype(np.int)
            image_frame_idx = [img_idx_list[idx_list[i]] for i in range(length)]
            return image_frame_idx

    def load_boxes(self, key, image_frame_idx, out_size):  # T,N,4
        key_bbox_list = [(key[0], str(key[1]), str(i).zfill(4)) for i in image_frame_idx]
        N = self.num_boxes
        T = self.length
        H, W = out_size
        boxes = []
        for key_bbox in key_bbox_list:
            person_idx_list = []
            for i, item in enumerate(self.boxes_dict[key_bbox]['box_label']):
                if item == 'person':
                    person_idx_list.append(i)
            tmp_bbox = []
            tmp_x1, tmp_y1, tmp_x2, tmp_y2 = 0, 0, 0, 0
            for idx, person_idx in enumerate(person_idx_list):
                if idx < N:
                    box = self.boxes_dict[key_bbox]['boxes'][person_idx]
                    box[:2] -= box[2:] / 2
                    x, y, w, h = box.tolist()
                    x = x * W
                    y = y * H
                    w = w * W
                    h = h * H
                    tmp_x1, tmp_y1, tmp_x2, tmp_y2 = x, y, x + w, y + h
                    tmp_bbox.append(torch.tensor([x, y, x + w, y + h]).unsqueeze(0))  # 1,4 x1,y1,x2,y2
            if len(person_idx_list) < N:
                step = len(person_idx_list)
                while step < N:
                    tmp_bbox.append(torch.tensor([tmp_x1, tmp_y1, tmp_x2, tmp_y2]).unsqueeze(0))  # 1,4
                    step += 1
            boxes.append(torch.cat(tmp_bbox).unsqueeze(0))  # 1,N,4
        boxes_tensor = torch.cat(boxes)
        return boxes_tensor

    def random_select_frames(self, video, image_frame_idx):
        length = self.length
        num_selected_frames = self.num_selected_frames
        select_list_per_clip = [i for i in range(16)]
        selected_frames_list = []
        selected_frames_idx = []
        for i in range(length // 10):
            random_sample_list = random.sample(select_list_per_clip, num_selected_frames)
            selected_frames_list.extend([video[10 * i + j].unsqueeze(0) for j in random_sample_list])
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in random_sample_list])
        selected_frames = torch.cat(selected_frames_list, dim=0)  # 540*t,C,H,W; t=num_selected_frames
        return selected_frames, selected_frames_idx

    def select_middle_frames(self, video, image_frame_idx):
        length = self.length
        num_selected_frames = self.num_selected_frames
        selected_frames_list = []
        selected_frames_idx = []
        for i in range(length // 10):
            sample_list = [16 // (num_selected_frames + 1) * (j + 1) - 1 for j in range(num_selected_frames)]
            selected_frames_list.extend([video[10 * i + j].unsqueeze(0) for j in sample_list])
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in sample_list])
        selected_frames = torch.cat(selected_frames_list, dim=0)  # 540*t,C,H,W; t=num_selected_frames
        return selected_frames, selected_frames_idx

    def random_select_idx(self, image_frame_idx):
        length = self.length
        num_selected_frames = self.num_selected_frames
        select_list_per_clip = [i for i in range(16)]
        selected_frames_idx = []
        for i in range(length // 10):
            random_sample_list = random.sample(select_list_per_clip, num_selected_frames)
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in random_sample_list])
        return selected_frames_idx

    def select_middle_idx(self, image_frame_idx):
        length = self.length
        num_selected_frames = self.num_selected_frames
        selected_frames_idx = []
        for i in range(length // 10):
            sample_list = [16 // (num_selected_frames + 1) * (j + 1) - 1 for j in range(num_selected_frames)]
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in sample_list])
        return selected_frames_idx

    def load_goat_data(self, data: dict, key: tuple):
        if self.args.use_goat:
            if self.args.use_formation:
                # use formation features
                data['formation_features'] = self.formation_features_dict[key]  # 540,1024 [Middle]
            elif self.args.use_bp:
                # use bp features
                file_name = key[0] + '_' + str(key[1]) + '.npy'
                bp_features_ori = torch.tensor(np.load(os.path.join(self.bp_feature_path, file_name)))  # T_ori,768
                if bp_features_ori.shape[0] == 768:
                    bp_features_ori = bp_features_ori.reshape(-1, 768)
                frames_path = os.path.join(self.data_path, key[0], str(key[1]))
                image_frame_idx = self.load_idx(frames_path)  # T,C,H,W
                if self.args.random_select_frames:
                    selected_frames_idx = self.random_select_idx(image_frame_idx)
                else:
                    selected_frames_idx = self.select_middle_idx(image_frame_idx)
                bp_features_list = [bp_features_ori[i].unsqueeze(0) for i in selected_frames_idx]  # [1,768]
                data['bp_features'] = torch.cat(bp_features_list, dim=0).to(torch.float32)  # 540,768
            elif self.args.use_self:
                data = data
            else:
                # use group features
                if self.args.use_cnn_features:
                    frames_path = os.path.join(self.data_path, key[0], str(key[1]))
                    image_frame_idx = self.load_idx(frames_path)  # T,C,H,W
                    if self.args.random_select_frames:
                        selected_frames_idx = self.random_select_idx(image_frame_idx)
                    else:
                        selected_frames_idx = self.select_middle_idx(image_frame_idx)
                    data['boxes'] = self.load_boxes(key, selected_frames_idx, self.out_size)  # 540*t,N,4
                    data['cnn_features'] = self.cnn_feature_dict[key].squeeze(0)
                else:
                    frames_path = os.path.join(self.data_path, key[0], str(key[1]))
                    video, image_frame_idx = self.load_video(frames_path)  # T,C,H,W
                    if self.args.random_select_frames:
                        selected_frames, selected_frames_idx = self.random_select_frames(video, image_frame_idx)
                    else:
                        selected_frames, selected_frames_idx = self.select_middle_frames(video, image_frame_idx)
                    data['boxes'] = self.load_boxes(key, selected_frames_idx, self.out_size)  # 540*t,N,4
                    data['video'] = selected_frames  # 540*t,C,H,W
        return data

    def __getitem__(self, ix):
        key = self.keys[ix]
        # build data
        data = {}
        data['final_score'] = self.annotations.get(key)[1]
        data['feature'] = self.feature_dict[key]  # 540,1024
        data['key'] = key
        data = self.load_goat_data(data, key)
        self.proc_label(data)
        return data

    def __len__(self):
        return len(self.keys)
