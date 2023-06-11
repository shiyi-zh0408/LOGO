import torch
import numpy as np
import os
import pickle
import random
import glob
# from os.path import join
from PIL import Image
from torchvideotransforms import video_transforms, volume_transforms
import pickle as pkl


class MTLPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        if args.use_i3d_bb:
            args.feature_path = args.i3d_feature_path
        elif args.use_swin_bb:
            args.feature_path = args.swin_feature_path
        else:
            args.feature_path = args.bpbb_feature_path
        self.args = args
        self.subset = subset
        self.transforms = transform
        # some flags
        self.usingDD = args.usingDD
        self.dive_number_choosing = args.dive_number_choosing
        # file path
        self.label_path = args.label_path
        self.split_path = args.train_split
        self.feature_path = args.feature_path
        self.cnn_feature_path = args.cnn_feature_path
        self.split = self.read_pickle(self.split_path)
        self.label_dict = self.read_pickle(self.label_path)
        self.feature_dict = self.read_pickle(self.feature_path)
        self.cnn_feature_dict = self.read_pickle(self.cnn_feature_path)
        self.formation_features_dict = pkl.load(open(args.formation_feature_path, 'rb'))
        self.bp_feature_path = args.bp_feature_path
        self.boxes_dict = pkl.load(open(args.boxes_path, 'rb'))
        self.data_root = args.data_root
        # setting
        self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
        self.voter_number = args.voter_number
        self.length = args.length
        self.img_size = args.img_size
        self.num_boxes = args.num_boxes
        self.out_size = args.out_size
        self.num_selected_frames = args.num_selected_frames
        # build difficulty dict ( difficulty of each action, the cue to choose exemplar)
        self.difficulties_dict = {}
        self.dive_number_dict = {}
        if self.subset == 'test':
            self.split_path_test = args.test_split
            self.split_test = self.read_pickle(self.split_path_test)
            self.difficulties_dict_test = {}
            self.dive_number_dict_test = {}
        self.preprocess()
        # transforms
        self.transforms = video_transforms.Compose([
            video_transforms.Resize(self.img_size),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # if self.usingDD:

        #     self.check()

        self.choose_list = self.split.copy()
        if self.subset == 'test':
            self.dataset = self.split_test
        else:
            self.dataset = self.split

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

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

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

    def preprocess(self):
        for item in self.split:
            difficulty = self.label_dict.get(item)[0]
            if self.difficulties_dict.get(difficulty) is None:
                self.difficulties_dict[difficulty] = []
            self.difficulties_dict[difficulty].append(item)

        if self.subset == 'test':
            for item in self.split_test:
                difficulty = self.label_dict.get(item)[0]
                if self.difficulties_dict_test.get(difficulty) is None:
                    self.difficulties_dict_test[difficulty] = []
                self.difficulties_dict_test[difficulty].append(item)

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
                frames_path = os.path.join(self.data_root, key[0], str(key[1]))
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
                    frames_path = os.path.join(self.data_root, key[0], str(key[1]))
                    image_frame_idx = self.load_idx(frames_path)  # T,C,H,W
                    if self.args.random_select_frames:
                        selected_frames_idx = self.random_select_idx(image_frame_idx)
                    else:
                        selected_frames_idx = self.select_middle_idx(image_frame_idx)
                    data['boxes'] = self.load_boxes(key, selected_frames_idx, self.out_size)  # 540*t,N,4
                    data['cnn_features'] = self.cnn_feature_dict[key].squeeze(0)
                else:
                    frames_path = os.path.join(self.data_root, key[0], str(key[1]))
                    video, image_frame_idx = self.load_video(frames_path)  # T,C,H,W
                    if self.args.random_select_frames:
                        selected_frames, selected_frames_idx = self.random_select_frames(video, image_frame_idx)
                    else:
                        selected_frames, selected_frames_idx = self.select_middle_frames(video, image_frame_idx)
                    data['boxes'] = self.load_boxes(key, selected_frames_idx, self.out_size)  # 540*t,N,4
                    data['video'] = selected_frames  # 540*t,C,H,W
        return data

    def delta(self):
        delta = []
        dataset = self.split.copy()
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                delta.append(
                    abs(
                        self.label_dict[dataset[i]][1] -
                        self.label_dict[dataset[j]][1]))

        return delta

    def __getitem__(self, index):
        key = self.dataset[index]
        data = {}
        if self.subset == 'test':
            # test phase
            data['feature'] = self.feature_dict[key]
            data['final_score'] = self.label_dict.get(key)[1]
            # DD---TYPE
            if self.label_dict.get(key)[0] == 'free':
                data['difficulty'] = 0
            elif self.label_dict.get(key)[0] == 'tech':
                data['difficulty'] = 1
            train_file_list = self.difficulties_dict[self.label_dict[key][0]]
            random.shuffle(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]
            # goat
            data = self.load_goat_data(data, key)

            # exemplar
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['feature'] = self.feature_dict[item]
                tmp['final_score'] = self.label_dict.get(item)[1]
                if self.label_dict.get(item)[0] == 'free':
                    tmp['difficulty'] = 0
                elif self.label_dict.get(item)[0] == 'tech':
                    tmp['difficulty'] = 1
                # goat
                tmp = self.load_goat_data(tmp, item)
                target_list.append(tmp)

            return data, target_list
        else:
            # train phase
            data['feature'] = self.feature_dict[key]
            data['final_score'] = self.label_dict.get(key)[1]
            if self.label_dict.get(key)[0] == 'free':
                data['difficulty'] = 0
            elif self.label_dict.get(key)[0] == 'tech':
                data['difficulty'] = 1
            # goat
            data = self.load_goat_data(data, key)

            file_list = self.difficulties_dict[self.label_dict[key][0]].copy()  # @
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(key))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            # sample 2
            target['feature'] = self.feature_dict[sample_2]
            target['final_score'] = self.label_dict.get(sample_2)[1]
            if self.label_dict.get(sample_2)[0] == 'free':
                target['difficulty'] = 0
            elif self.label_dict.get(sample_2)[0] == 'tech':
                target['difficulty'] = 1
            # goat
            target = self.load_goat_data(target, sample_2)

            return data, target

    def __len__(self):
        return len(self.dataset)
