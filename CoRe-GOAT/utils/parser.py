import os
import yaml
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type = str, choices=['MTL', 'Seven'], help = 'dataset')
    parser.add_argument('--fix_bn', type=bool, default=True)
    parser.add_argument('--dive_number_choosing', type=bool, default=False)
    parser.add_argument('--usingDD', type=bool, default=False)
    parser.add_argument('--resume', action='store_true', default=False ,help = 'autoresume training from exp dir(interrupted by accident)')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--optimizer', type = str, default='Adam', help = '')
    parser.add_argument('--Seven_cls', type = int, default=1, choices=[1,2,3,4,5,6], help = 'class idx in Seven')
    parser.add_argument('--bs_train', type=int, default=1, help = 'batch size of training')
    parser.add_argument('--bs_test', type=int, default=1, help = 'batch size of testing')
    parser.add_argument('--workers', type=int, default=24, help = 'number of workers')
    parser.add_argument('--step_per_update', type=int, default=2, help = 'step_per_update')
    parser.add_argument('--max_epoch', type=int, default=200, help = 'epoch number')
    parser.add_argument('--RT_depth', type=int, default=5, help = '')
    parser.add_argument('--score_range', type=int, default=100, help = '')
    parser.add_argument('--voter_number', type=int, default=10, help = '')
    parser.add_argument('--seed', type=int, default=42, help = '')
    parser.add_argument('--weight_decay', type=int, default=0, help = '')
    parser.add_argument('--temporal_shift_min', type=int, default=-3, help = '')
    parser.add_argument('--temporal_shift_max', type=int, default=3, help = '')
    parser.add_argument('--print_freq', type=int, default=40, help = '')


    ###################################################################################################################
    parser.add_argument('--length', type=int, help='length of videos', default=5406)
    parser.add_argument('--lr', type=float, help='learning rate', default=3e-3)
    parser.add_argument('--lr_factor', type=float, help='learning rate factor', default=0.01)
    parser.add_argument('--img_size', type=tuple, help='input image size', default=(224, 224))
    parser.add_argument('--out_size', type=tuple, help='output image size', default=(25, 25))
    parser.add_argument('--crop_size', type=tuple, help='RoiAlign image size', default=(5, 5))
    parser.add_argument('--label_path', type=str, help='path of annotation file', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/Anno_result/anno_dict.pkl')
    parser.add_argument('--train_split', type=str, help='', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/Anno_result/train_split3.pkl')
    parser.add_argument('--test_split', type=str, help='', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/Anno_result/test_split3.pkl')
    parser.add_argument('--boxes_path', type=str, help='path of boxes annotation file', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/DINO/ob_result_new.pkl')
    # backbone features path
    parser.add_argument('--i3d_feature_path', type=str, help='path of i3d feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/video_feature_dict.pkl')
    parser.add_argument('--swin_feature_path', type=str, help='path of swin feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/video-swin-features/swin_features_dict_new.pkl')
    parser.add_argument('--bpbb_feature_path', type=str, help='path of bridge-prompt feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/bpbb_features_540.pkl')
    # attention features path
    parser.add_argument('--cnn_feature_path', type=str, help='path of cnn feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/Inceptionv3/inception_feature_dict.pkl')
    parser.add_argument('--bp_feature_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/bp_features', help='bridge prompt feature path')
    parser.add_argument('--formation_feature_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/formation_features_middle_1.pkl', help='formation feature path')
    # others
    parser.add_argument('--data_root', type=str, help='root of dataset', default='/mnt/petrelfs/daiwenxun/AS-AQA/Video_result')
    parser.add_argument('--num_boxes', type=int, help='boxes number of each frames', default=8)
    parser.add_argument('--num_selected_frames', type=int, help='number of selected frames per 16 frames', default=1)
    parser.add_argument('--stage1_model_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/Group-AQA-Distributed/ckpts/STAGE1_256frames_rho0.3257707338254451_(224, 224)_(25, 25)_loss82.48323059082031.pth', help='stage1_model_path')

    # [BOOL]
    # bool for attention mode[GOAT / BP / FORMATION / SELF]
    parser.add_argument('--use_goat', type=int, help='whether to use group-aware-attention', default=1)
    parser.add_argument('--use_bp', type=int, help='whether to use bridge prompt features', default=0)
    parser.add_argument('--use_formation', type=int, help='whether to use formation features', default=0)
    parser.add_argument('--use_self', type=int, help='whether to use self attention', default=0)
    # bool for backbone[I3D / SWIN / BP]
    parser.add_argument('--use_i3d_bb', type=int, help='whether to use i3d as backbone', default=1)
    parser.add_argument('--use_swin_bb', type=int, help='whether to use swin as backbone', default=0)
    parser.add_argument('--use_bp_bb', type=int, help='whether to use bridge-prompt as backbone', default=0)
    # bool for others
    parser.add_argument('--train_backbone', type=int, help='whether to train backbone', default=0)
    parser.add_argument('--use_gcn', type=int, help='whether to use gcn', default=1)
    parser.add_argument('--warmup', type=int, help='whether to warm up', default=1)
    parser.add_argument('--random_select_frames', type=int, help='whether to select frames randomly', default=1)
    parser.add_argument('--use_multi_gpu', type=int, help='whether to use multi gpus', default=1)
    parser.add_argument('--gcn_temporal_fuse', type=int, help='whether to fuse temporal node before gcn', default=0)
    parser.add_argument('--use_cnn_features', type=int, help='whether to use pretrained cnn features', default=1)

    parser.add_argument('--train_dropout_prob', type=float, default=0.3, help='train_dropout_prob')

    # log
    parser.add_argument('--exp_name', type=str, default='goat', help='experiment name')
    parser.add_argument('--result_path', type=str, default='result/result_new.csv', help='result log path')

    # attention
    parser.add_argument('--num_heads', type=int, default=8, help='number of self-attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--linear_dim', type=int, default=1024, help='dimension of query and key')
    parser.add_argument('--attn_drop', type=float, default=0., help='drop prob of attention layer')

    # fixed parameters
    parser.add_argument('--emb_features', type=int, default=1056, help='output feature map channel of backbone')
    parser.add_argument('--num_features_boxes', type=int, default=1024, help='dimension of features of each box')
    parser.add_argument('--num_features_relation', type=int, default=256, help='dimension of embedding phi(x) and theta(x) [Embedded Dot-Product]')
    parser.add_argument('--num_features_gcn', type=int, default=1024, help='dimension of features of each node')
    parser.add_argument('--num_graph', type=int, default=16, help='number of graphs')
    parser.add_argument('--gcn_layers', type=int, default=1, help='number of gcn layers')
    parser.add_argument('--pos_threshold', type=float, default=0.2, help='threshold for distance mask')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    ###################################################################################################################

    args = parser.parse_args()

    if args.test:
        if args.ckpts is None:
            raise RuntimeError('--ckpts should not be None when --test is activate')

    if args.benchmark == 'Seven':
        print(f'Using CLASS idx {args.Seven_cls}')
        args.class_idx = args.Seven_cls
    return args
