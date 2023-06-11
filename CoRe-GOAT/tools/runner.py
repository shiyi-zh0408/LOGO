import numpy as np
import torch
import torch.nn as nn

from scipy import stats
from tools import builder, helper
from utils import misc
import time
from models.cnn_model import GCNnet_artisticswimming
from models.group_aware_attention import Encoder_Blocks
from utils.multi_gpu import *
from models.cnn_simplified import GCNnet_artisticswimming_simplified
from models.linear_for_bp import Linear_For_Backbone
from thop import profile


def test_net(args):
    print('Tester start ... ')
    train_dataset, test_dataset = builder.dataset_builder(args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)
    base_model, regressor = builder.model_builder(args)
    # load checkpoints
    builder.load_model(base_model, regressor, args)

    # if using RT, build a group
    group = builder.build_group(train_dataset, args)

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # base_model = base_model.cuda()
        regressor = regressor.cuda()
        torch.backends.cudnn.benchmark = True

    #  DP
    # base_model = nn.DataParallel(base_model)
    regressor = nn.DataParallel(regressor)

    test(base_model, regressor, test_dataloader, group, args)


def run_net(args):
    if is_main_process():
        print('Trainer start ... ')
    # build dataset
    train_dataset, test_dataset = builder.dataset_builder(args)
    if args.use_multi_gpu:
        train_dataloader = build_dataloader(train_dataset,
                                            batch_size=args.bs_train,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            persistent_workers=True,
                                            seed=set_seed(args.seed))
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs_train,
                                                       shuffle=False, num_workers=int(args.workers),
                                                       pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)

    # Set data position
    device = get_device()

    # build model
    base_model, regressor = builder.model_builder(args)

    input1 = torch.randn(2, 2049)
    flops, params = profile(regressor, inputs=(input1, ))
    print(f'[regressor]flops: ', flops, 'params: ', params)

    # Set models and optimizer(depend on whether to use goat)
    if args.use_goat:
        if args.use_cnn_features:
            gcn = GCNnet_artisticswimming_simplified(args)

            input1 = torch.randn(1, 540, 8, 1024)
            input2 = torch.randn(1, 540, 8, 4)
            flops, params = profile(gcn, inputs=(input1, input2))
            print(f'[GCNnet_artisticswimming_simplified]flops: ', flops, 'params: ', params)
        else:
            gcn = GCNnet_artisticswimming(args)
            gcn.loadmodel(args.stage1_model_path)
        attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers, args.attn_drop)
        linear_bp = Linear_For_Backbone(args)
        optimizer = torch.optim.Adam([
            {'params': gcn.parameters(), 'lr': args.lr * args.lr_factor},
            {'params': regressor.parameters()},
            {'params': linear_bp.parameters()},
            {'params': attn_encoder.parameters()}
        ], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.use_multi_gpu:
            wrap_model(gcn, distributed=args.distributed)
            wrap_model(attn_encoder, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
            wrap_model(regressor, distributed=args.distributed)
        else:
            gcn = gcn.to(device=device)
            attn_encoder = attn_encoder.to(device=device)
            linear_bp = linear_bp.to(device=device)
            regressor = regressor.to(device=device)
    else:
        gcn = None
        attn_encoder = None
        linear_bp = Linear_For_Backbone(args)
        optimizer = torch.optim.Adam([{'params': regressor.parameters()}, {'params': linear_bp.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.use_multi_gpu:
            wrap_model(regressor, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
        else:
            regressor = regressor.to(device=device)
            linear_bp = linear_bp.to(device=device)

    if args.warmup:
        num_steps = len(train_dataloader) * args.max_epoch
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    # if using RT, build a group
    group = builder.build_group(train_dataset, args)
    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    # parameter setting
    start_epoch = 0
    global epoch_best, rho_best, L2_min, RL2_min
    epoch_best = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    # resume ckpts
    if args.resume:
        start_epoch, epoch_best, rho_best, L2_min, RL2_min = \
            builder.resume_train(base_model, regressor, optimizer, args)
        print('resume ckpts @ %d epoch( rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (
            start_epoch - 1, rho_best, L2_min, RL2_min))

    #  DP
    # regressor = nn.DataParallel(regressor)
    # if args.use_goat:
    #     gcn = nn.DataParallel(gcn)
    #     attn_encoder = nn.DataParallel(attn_encoder)

    # loss
    mse = nn.MSELoss().cuda()
    nll = nn.NLLLoss().cuda()

    # trainval

    # training
    for epoch in range(start_epoch, args.max_epoch):
        if args.use_multi_gpu:
            train_dataloader.sampler.set_epoch(epoch)
        true_scores = []
        pred_scores = []
        num_iter = 0
        # base_model.train()  # set model to training mode
        regressor.train()
        linear_bp.train()
        if args.use_goat:
            gcn.train()
            attn_encoder.train()
        # if args.fix_bn:
        #     base_model.apply(misc.fix_bn)  # fix bn
        for idx, (data, target) in enumerate(train_dataloader):
            # break
            num_iter += 1
            opti_flag = False

            true_scores.extend(data['final_score'].numpy())
            # data preparing
            # featue_1 is the test video ; video_2 is exemplar
            if args.benchmark == 'MTL':
                feature_1 = data['feature'].float().cuda()  # N, C, T, H, W
                if args.usingDD:
                    label_1 = data['completeness'].float().reshape(-1, 1).cuda()
                    label_2 = target['completeness'].float().reshape(-1, 1).cuda()
                else:
                    label_1 = data['final_score'].float().reshape(-1, 1).cuda()
                    label_2 = target['final_score'].float().reshape(-1, 1).cuda()
                if not args.dive_number_choosing and args.usingDD:
                    assert (data['difficulty'] == target['difficulty']).all()
                diff = data['difficulty'].float().reshape(-1, 1).cuda()
                feature_2 = target['feature'].float().cuda()  # N, C, T, H, W

            else:
                raise NotImplementedError()

            # forward
            if num_iter == args.step_per_update:
                num_iter = 0
                opti_flag = True

            helper.network_forward_train(base_model, regressor, pred_scores, feature_1, label_1, feature_2, label_2,
                                         diff, group, mse, nll, optimizer, opti_flag, epoch, idx + 1,
                                         len(train_dataloader), args, data, target, gcn, attn_encoder, device, linear_bp)

            if args.warmup:
                lr_scheduler.step()

        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        if is_main_process():
            print('[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f' % (
                epoch, rho, L2, RL2, optimizer.param_groups[0]['lr']))

        if is_main_process():
            validate(base_model, regressor, test_dataloader, epoch, optimizer, group, args, gcn, attn_encoder, device, linear_bp)
            # helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min,
            #                        'last',
            #                        args)
            print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (
                epoch, rho_best, L2_min, RL2_min))
        # scheduler lr
        if scheduler is not None:
            scheduler.step()


# TODO: 修改以下所有;修改['difficulty'].float
def validate(base_model, regressor, test_dataloader, epoch, optimizer, group, args, gcn, attn_encoder, device, linear_bp):
    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best, rho_best, L2_min, RL2_min
    true_scores = []
    pred_scores = []
    # base_model.eval()  # set model to eval mode
    regressor.eval()
    linear_bp.eval()
    if args.use_goat:
        gcn.eval()
        attn_encoder.eval()
    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()
        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()
            true_scores.extend(data['final_score'].numpy())
            # data prepare
            if args.benchmark == 'MTL':
                feature_1 = data['feature'].float().cuda()  # N, C, T, H, W
                if args.usingDD:
                    label_2_list = [item['completeness'].float().reshape(-1, 1).cuda() for item in target]
                else:
                    label_2_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
                diff = data['difficulty'].float().reshape(-1, 1).cuda()
                feature_2_list = [item['feature'].float().cuda() for item in target]
                # check
                if not args.dive_number_choosing and args.usingDD:
                    for item in target:
                        assert (diff == item['difficulty'].reshape(-1, 1).cuda()).all()
            else:
                raise NotImplementedError()
            helper.network_forward_test(base_model, regressor, pred_scores, feature_1, feature_2_list, label_2_list,
                                        diff, group, args, data, target, gcn, attn_encoder, device, linear_bp)
            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d][%d/%d] \t Batch_time %.6f \t Data_time %.6f '
                      % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        if L2_min > L2:
            L2_min = L2
        if RL2_min > RL2:
            RL2_min = RL2
        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            print('-----New best found!-----')
            # helper.save_outputs(pred_scores, true_scores, args)
            # helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min,
            #                        'best', args)
        if epoch == args.max_epoch - 1:
            log_best(rho_best, RL2_min, epoch_best, args)

        print('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % (epoch, rho, L2, RL2))


def test(base_model, regressor, test_dataloader, group, args, gcn, attn_encoder, device):
    global use_gpu
    true_scores = []
    pred_scores = []
    # base_model.eval()  # set model to eval mode
    regressor.eval()
    if args.use_goat:
        gcn.eval()
        attn_encoder.eval()
    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()
        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()
            true_scores.extend(data['final_score'].numpy())
            # data prepare
            if args.benchmark == 'MTL':
                featue_1 = data['feature'].float().cuda()  # N, C, T, H, W
                if args.usingDD:
                    label_2_list = [item['completeness'].float().reshape(-1, 1).cuda() for item in target]
                else:
                    label_2_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
                diff = data['difficulty'].float().reshape(-1, 1).cuda()
                feature_2_list = [item['feature'].float().cuda() for item in target]
                # check
                if not args.dive_number_choosing and args.usingDD:
                    for item in target:
                        assert (diff == item['difficulty'].float().reshape(-1, 1).cuda()).all()
            elif args.benchmark == 'Seven':
                featue_1 = data['feature'].float().cuda()  # N, C, T, H, W
                feature_2_list = [item['feature'].float().cuda() for item in target]
                label_2_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
                diff = None
            else:
                raise NotImplementedError()
            helper.network_forward_test(base_model, regressor, pred_scores, featue_1, feature_2_list, label_2_list,
                                        diff, group, args, data, target, gcn, attn_encoder, device)
            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d] \t Batch_time %.2f \t Data_time %.2f '
                      % (batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        print('[TEST] correlation: %.6f, L2: %.6f, RL2: %.6f' % (rho, L2, RL2))
