import torch.nn as nn
import torch
from .i3d import I3D
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class I3D_backbone(nn.Module):
    def __init__(self, I3D_class):
        super(I3D_backbone, self).__init__()
        print('Using I3D backbone')
        self.backbone = I3D(num_classes=I3D_class, modality='rgb', dropout_prob=0.5)

    def load_pretrain(self, I3D_ckpt_path):
        self.backbone.load_state_dict(torch.load(I3D_ckpt_path))
        print('loading ckpt done')

    def get_feature_dim(self):
        return self.backbone.get_logits_dim()

    def forward(self, target, exemplar, is_train, label, theta):
        # spatiotemporal feature
        total_video = torch.cat((target, exemplar), 0)  # 2N C H W
        start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]
        video_pack = torch.cat([total_video[:, :, i: i + 16] for i in start_idx])  # 10*2N, c, 16, h, w
        total_feature = self.backbone(video_pack).reshape(10, len(total_video), -1).transpose(0, 1)  # 2N * 10 * 1024
        total_feature = total_feature.mean(1)
        # 2N * 1024

        feature_1 = total_feature[:total_feature.shape[0] // 2]
        feature_2 = total_feature[total_feature.shape[0] // 2:]
        # N * 1024
        if is_train:
            combined_feature_1 = torch.cat((feature_1, feature_2, label[0] / theta), 1)  # 1 is exemplar N * 2049
            combined_feature_2 = torch.cat((feature_2, feature_1, label[1] / theta), 1)  # 2 is exemplar N * 2049
            return combined_feature_1, combined_feature_2
        else:
            combined_feature = torch.cat((feature_2, feature_1, label[0] / theta), 1)  # 2 is exemplar
            return combined_feature


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
