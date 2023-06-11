import torch.nn as nn
import torch
from .i3d import I3D
import logging
import torch.nn.functional as F
import torchvision.models as models


class I3D_backbone(nn.Module):
    def __init__(self, I3D_class):
        super(I3D_backbone, self).__init__()
        print('Using I3D backbone')
        self.backbone = I3D(num_classes=I3D_class, modality='rgb', dropout_prob=0.5)

    def load_pretrain(self, I3D_ckpt_path):
        try:
            self.backbone.load_state_dict(torch.load(I3D_ckpt_path))
            print('loading ckpt done')
        except:
            logging.info('Ckpt path {} do not exists'.format(I3D_ckpt_path))
            pass

    def forward(self, video_1, video_2):

        total_video = torch.cat((video_1, video_2), 0)
        start_idx = list(range(0, 90, 10))
        video_pack = torch.cat([total_video[:, :, i: i + 16] for i in start_idx])
        total_feamap, total_feature = self.backbone(video_pack)
        Nt, C, T, H, W = total_feamap.size()

        total_feature = total_feature.reshape(len(start_idx), len(total_video), -1).transpose(0, 1)  # (2N, 9, 1024)
        total_feamap = total_feamap.reshape(len(start_idx), len(total_video), C, T, H, W).transpose(0,
                                                                                                    1)  # (2N, 9, 1024, 2, 4, 4)

        com_feature_12 = torch.cat(
            (total_feature[:total_feature.shape[0] // 2], total_feature[total_feature.shape[0] // 2:]),
            2)  # (N, 9, 2048)
        com_feamap_12 = torch.cat(
            (total_feamap[:total_feamap.shape[0] // 2], total_feamap[total_feamap.shape[0] // 2:]),
            2)  # (N, 9, 2048. 2, 4, 4)
        return com_feature_12, com_feamap_12


class MyInception_v3(nn.Module):
    def __init__(self, transform_input=False, pretrained=False):
        super(MyInception_v3, self).__init__()
        self.transform_input = transform_input
        inception = models.inception_v3(pretrained=pretrained)

        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e

    def forward(self, x):
        outputs = []

        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        outputs.append(x)

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        outputs.append(x)

        return outputs  # cuda:0


class MyVGG16(nn.Module):
    def __init__(self, pretrained=False):
        super(MyVGG16, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)

        self.features = vgg.features

    def forward(self, x):
        x = self.features(x)
        return [x]


class MyVGG19(nn.Module):
    def __init__(self, pretrained=False):
        super(MyVGG19, self).__init__()

        vgg = models.vgg19(pretrained=pretrained)

        self.features = vgg.features

    def forward(self, x):
        x = self.features(x)
        return [x]
