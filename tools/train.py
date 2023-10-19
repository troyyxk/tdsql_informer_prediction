# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by BgZhang
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import ptvsd
# ptvsd.enable_attach(address=('192.168.3.250',5678))
# ptvsd.wait_for_attach()

import argparse
import os
import pprint
import shutil
import sys
import json
import time
import numpy as np
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
# from prefetch_generator import BackgroundGenerator


import _init_paths
import models
from config import read_config
from core.function import train
from core.function import validate
from trick_dl.earlystop import EarlyStopping
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.dataset import Dataset_TDSQL


# # dataload底层原生实现较慢，改为这种形式
# class DataLoaderX(torch.utils.data.DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())


# 设置随机数种子， 酌情使用
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 解析对应的配置文件
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')
    parser.add_argument('--cfg', help='experiment configure file name',
                        default="./experiments/xxxx.json", required=True, type=str)
    args = parser.parse_args()
    cfg = read_config(args)  # 这一步中是为了将parser.add_argument中的参数加入到json配置文件中
    return cfg, args


def main():
    config, args = parse_args()

    # 设置随机数种子
    # setup_seed(config['TRAIN']["SEED"])

    logger, final_output_dir, tb_log_dir, _ = create_logger(
        config, args.cfg, 'train')  # 生成存放模型和日志的文件夹
    logger.info(pprint.pformat(args))  # 打印日志
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config['CUDNN']['BENCHMARK']
    torch.backends.cudnn.deterministic = config['CUDNN']['DETERMINISTIC']
    # cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    torch.backends.cudnn.enabled = config['CUDNN']['ENABLED']

    # TODO 这里要手动改一下
    model = eval('models.' + 'model.Informer')(
        enc_in=config["MODEL"]["ENC_IN"],
        dec_in=config["MODEL"]["DEC_IN"],
        c_out=config["MODEL"]["C_OUT"],
        seq_len=config["DATASET"]["SEQ_LEN"],
        label_len=config["DATASET"]["LABEL_LEN"],
        out_len=config["DATASET"]["PRED_LEN"],
        factor=config["MODEL"]["FACTOR"],
        d_model=config["MODEL"]["D_MODEL"],
        n_heads=config["MODEL"]["N_HEADS"],
        e_layers=config["MODEL"]["E_LAYERS"],
        d_layers=config["MODEL"]["D_LAYERS"],
        d_ff=config["MODEL"]["D_FF"],
        dropout=config["MODEL"]["DROPOUT"],
        attn=config["MODEL"]["ATTN"],
        embed=config["MODEL"]["EMBED"],
        freq=config["DATASET"]["FREQ"],
        activation=config["MODEL"]["ACTIVATION"],
        output_attention=config["MODEL"]["OUTPUT_ATTENTION"],
        distil=config["MODEL"]["DISTIL"],
        mix=config["MODEL"]["MIX"]
    )  # 模型的定义

    # # 以下两行均可以注释，非必要
    # 用于输出模型的参数量
    # dump_input = torch.rand((1, 3, config['MODEL']['IMAGE_SIZE'][1], config['MODEL']['IMAGE_SIZE'][0]))  # 定义随机数据
    # logger.info(get_model_summary(model, dump_input))  # 测试模型输入是否正常，同时打印模型的具体参数个数和计算量等

    # 新建文件夹，复制模型的配置文件到该文件夹下
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    os.makedirs(models_dst_dir, exist_ok=False)
    # 保存配置文件
    with open(os.path.join(models_dst_dir, os.path.basename(args.cfg)), 'w') as f:
        json.dump(config, f)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0
    }
    # for eval_dir_name in config["TEST"]["EVAL_SET"]:
    #     writer_dict[f'{eval_dir_name}_global_steps'] = 0

    gpus = list(config['GPUS'])
    # 表示同时使用多个GPU，GPU的序号配置在json文件中
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    # model.cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss().cuda()

    optimizer = get_optimizer(config, model)  # 选择优化器

    best_perf = 100.0
    best_model = False
    last_epoch = config['TRAIN']['BEGIN_EPOCH']  # 表示应该从那个epoch开始
    if config['TRAIN']['RESUME']:
        # 表示从已有的模型加载并继续训练
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            # model.module.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True

    # lr的具体下降策略，多少个epoch下降一次
    assert config['TRAIN']['LR_SCHEDULER']['LR_SCHEDULER'] in ['MultiStepLR', 'StepLR', 'ExponentialLR',
                                                               'EARLYSTOPPING'], f"{config['TRAIN']['LR_SCHEDULER']} not in this code, you should add it"
    if config['TRAIN']['LR_SCHEDULER']['LR_SCHEDULER'] == 'MultiStepLR':
        # 多级下降策略，LR_STEP为一个list，每到list中的一个epoch时LR乘以LR_FACTOR倍
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config['TRAIN']['LR_SCHEDULER']['MultiStepLR'][
                'LR_STEP'], config['TRAIN']['LR_SCHEDULER']['MultiStepLR']['LR_FACTOR'],
            last_epoch-1
        )  # 这里的config['TRAIN']['LR_STEP']是一个list
    elif config['TRAIN']['LR_SCHEDULER']['LR_SCHEDULER'] == 'StepLR':
        # 单级下降策略，LR_STEP为一个值，epoch到该值时下降LR_FACTOR倍
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config['TRAIN']['LR_SCHEDULER']['StepLR']['LR_STEP'], config['TRAIN']['LR_SCHEDULER']['StepLR']['LR_FACTOR'],
            last_epoch-1
        )  # 这里的config['TRAIN']['LR_STEP']是一个值
    elif config['TRAIN']['LR_SCHEDULER']['LR_SCHEDULER'] == 'ExponentialLR':
        # 指数下降策略
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=config['TRAIN']['LR_SCHEDULER']['ExponentialLR']['EXPONENTIAL_LR_GAMMA'],
            last_epoch=last_epoch-1
        )  # verbose (bool) – If True, prints a message to stdout for each update. Default: False.
    elif config['TRAIN']['LR_SCHEDULER']['LR_SCHEDULER'] == 'EARLYSTOPPING':
        # earlystop下降策略，即多少个epoch测试集准确率不提升，则学习率下降
        lr_scheduler = EarlyStopping(
            optimizer=optimizer,
            patience=config['TRAIN']['LR_SCHEDULER']['EARLYSTOPPING']['PATIENCE'],
            delta=config['TRAIN']['LR_SCHEDULER']['EARLYSTOPPING']['DELTA'],
            min_lr=config['TRAIN']['LR_SCHEDULER']['EARLYSTOPPING']['MIN_LR'],
            flag=config['TRAIN']['LR_SCHEDULER']['EARLYSTOPPING']['FLAG']
        )
    else:
        print('lr_scheduler error')
        exit(0)

    # Data loading code
    train_file_path = os.path.join(
        config['DATASET']['ROOT_PATH'], config['DATASET']['TRAIN_FILE_NAME'])  # 训练文件
    val_file_path = os.path.join(
        config['DATASET']['ROOT_PATH'], config['DATASET']['VAL_FILE_NAME'])  # 测试文件

    # TODO
    train_dataset = Dataset_TDSQL(
        file_path=train_file_path,
        con_name_select=config["DATASET"]["COL_NAME_SQL_AMOUNT"],
        flag='train',
        inverse=False,
        timeenc=config["DATASET"]["TIMEENC"],
        freq=config["DATASET"]["FREQ"],
        seq_len=config["DATASET"]["SEQ_LEN"],
        label_len=config["DATASET"]["LABEL_LEN"],
        pred_len=config["DATASET"]["PRED_LEN"]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['TRAIN']['BATCH_SIZE_PER_GPU'] *
        len(gpus),  # 实际上的batchsize是原有的值乘以gpu数
        shuffle=True,
        num_workers=config['WORKERS'],  # debug的话,这个值要设成0
        pin_memory=True,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        Dataset_TDSQL(
            file_path=val_file_path,
            con_name_select=config["DATASET"]["COL_NAME_SQL_AMOUNT"],
            flag='val',
            inverse=False,
            timeenc=config["DATASET"]["TIMEENC"],
            freq=config["DATASET"]["FREQ"],
            seq_len=config["DATASET"]["SEQ_LEN"],
            label_len=config["DATASET"]["LABEL_LEN"],
            pred_len=config["DATASET"]["PRED_LEN"]
        ),
        batch_size=config['TRAIN']['BATCH_SIZE_PER_GPU']*len(gpus),
        shuffle=False,
        num_workers=config['WORKERS'],
        pin_memory=True,
        drop_last=True
    )

    for epoch in range(last_epoch, config['TRAIN']['END_EPOCH']):
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer,
              epoch, final_output_dir, tb_log_dir, writer_dict)

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,
                                  final_output_dir, tb_log_dir, writer_dict, prefix='valid')

        # 这里要注意一下
        if perf_indicator < best_perf:
            best_perf = perf_indicator
            best_model = True
            print("model save ###")
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config['MODEL']['NAME'],
            'state_dict': model.module.state_dict(),  # 'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

        if config['TRAIN']['LR_SCHEDULER']['LR_SCHEDULER'] == 'EARLYSTOPPING':
            # earlystop的lr调整策略
            lr_scheduler(perf_indicator)
            if lr_scheduler.early_stop:
                cont_train = lr_scheduler.adjust_learning_rate()
                if cont_train:
                    logger.info(
                        "Learning rate dropped by 10, continue training...")
                else:
                    logger.info('Early stopping.')
                    break
        else:
            # 非earlystop的lr调整策略
            lr_scheduler.step()
        logger.info('\n')

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    logger.info('This model\'s best val best_perf is {:.4f}'.format(best_perf))
    # torch.save(model.module.state_dict(), final_model_state_file)
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
