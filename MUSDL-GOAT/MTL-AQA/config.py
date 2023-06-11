import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_info', type=str, help='info that will be displayed when logging', default='Exp1')
    parser.add_argument('--std', type=float, help='standard deviation for gaussian distribution learning', default=5)
    parser.add_argument('--save', action='store_true', help='if set true, save the best model', default=False)
    parser.add_argument('--type', type=str, help='type of the model: USDL or MUSDL', choices=['USDL', 'MUSDL'], default='USDL')
    parser.add_argument('--temporal_aug', type=int, help='the maximum of random temporal shift, ranges from 0 to 6', default=6)
    parser.add_argument('--gpu', type=str, help='id of gpu device(s) to be used', default='2,3')
    parser.add_argument('--split', type=int, help='number of training epochs', default=3)

    # [BASIC]
    parser.add_argument('--num_epochs', type=int, help='number of training epochs', default=200)
    parser.add_argument('--train_batch_size', type=int, help='batch size for training phase', default=1)
    parser.add_argument('--test_batch_size', type=int, help='batch size for test phase', default=1)
    parser.add_argument('--seed', type=int, help='manual seed', default=42)
    parser.add_argument('--num_workers', type=int, help='number of subprocesses for dataloader', default=12)
    parser.add_argument('--lr', type=float, help='learning rate', default=3e-4)
    parser.add_argument('--weight_decay', type=float, help='L2 weight decay', default=1e-5)

    # [GOAT SETTING BELOW]
    # [CNN]
    parser.add_argument('--length', type=int, help='length of videos', default=5406)
    parser.add_argument('--img_size', type=tuple, help='input image size', default=(224, 224))
    parser.add_argument('--out_size', type=tuple, help='output image size', default=(25, 25))
    parser.add_argument('--crop_size', type=tuple, help='RoiAlign image size', default=(5, 5))

    # [GCN]
    parser.add_argument('--num_boxes', type=int, help='boxes number of each frames', default=8)
    parser.add_argument('--num_selected_frames', type=int, help='number of selected frames per 16 frames', default=1)

    # [PATH]
    parser.add_argument('--data_path', type=str, help='root of dataset', default='/mnt/petrelfs/daiwenxun/AS-AQA/Video_result')
    parser.add_argument('--anno_path', type=str, help='path of annotation file', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/Anno_result/anno_dict.pkl')
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
    parser.add_argument('--stage1_model_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/Group-AQA-Distributed/ckpts/STAGE1_256frames_rho0.3257707338254451_(224, 224)_(25, 25)_loss82.48323059082031.pth', help='stage1_model_path')
    parser.add_argument('--train_dropout_prob', type=float, default=0.3, help='train_dropout_prob')

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

    # [LOG]
    parser.add_argument('--exp_name', type=str, default='goat', help='experiment name')
    parser.add_argument('--result_path', type=str, default='result/result_new.csv', help='result log path')

    # [ATTENTION]
    parser.add_argument('--num_heads', type=int, default=8, help='number of self-attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--linear_dim', type=int, default=1024, help='dimension of query and key')
    parser.add_argument('--attn_drop', type=float, default=0., help='drop prob of attention layer')

    # [FIXED PARAMETERS]
    parser.add_argument('--emb_features', type=int, default=1056, help='output feature map channel of backbone')
    parser.add_argument('--num_features_boxes', type=int, default=1024, help='dimension of features of each box')
    parser.add_argument('--num_features_relation', type=int, default=256, help='dimension of embedding phi(x) and theta(x) [Embedded Dot-Product]')
    parser.add_argument('--num_features_gcn', type=int, default=1024, help='dimension of features of each node')
    parser.add_argument('--num_graph', type=int, default=16, help='number of graphs')
    parser.add_argument('--gcn_layers', type=int, default=1, help='number of gcn layers')
    parser.add_argument('--pos_threshold', type=float, default=0.2, help='threshold for distance mask')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
