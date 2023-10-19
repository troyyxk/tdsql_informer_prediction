from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys
import json
import time
import numpy as np
import random
import matplotlib.pyplot as plt

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
from core.function import draw_validate
from utils.utils import get_optimizer
from utils.utils import create_logger
from utils.dataset import Dataset_TDSQL

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
        config, args.cfg, 'valid')  # 生成存放模型和日志的文件夹
    logger.info(pprint.pformat(args))  # 打印日志
    logger.info(pprint.pformat(config))
    
    # save the npz file
    path_test_npz = os.path.join(final_output_dir, 'test_npz_file')
    os.makedirs(
        path_test_npz,
        exist_ok=True
    )
    
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

    if config['TEST']['MODEL_FILE']:
        logger.info('=> loading model from {}'.format(config['TEST']['MODEL_FILE']))
        model.load_state_dict(torch.load(config['TEST']['MODEL_FILE']))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'model_best.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    gpus = list(config['GPUS'])
    # 表示同时使用多个GPU，GPU的序号配置在json文件中
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    # model.cuda()


    # Data loading code
    val_file_path = os.path.join(
        config['DATASET']['ROOT_PATH'], config['DATASET']['VAL_FILE_NAME'])  # 测试文件


    valid_loader = torch.utils.data.DataLoader(
        Dataset_TDSQL(
            file_path=val_file_path,
            con_name_select=config["DATASET"]["COL_NAME_SQL_AMOUNT"],
            flag='test',
            inverse=False,
            timeenc=config["DATASET"]["TIMEENC"],
            freq=config["DATASET"]["FREQ"],
            seq_len=config["DATASET"]["SEQ_LEN"],
            label_len=config["DATASET"]["LABEL_LEN"],
            pred_len=config["DATASET"]["PRED_LEN"]
        ),
        batch_size=config['TEST']['BATCH_SIZE_PER_GPU']*len(gpus),
        shuffle=False,
        num_workers=config['WORKERS'],
        pin_memory=True,
        drop_last=True
    )

    draw_validate(config, valid_loader, model, path_test_npz)
    
    preds = np.load(os.path.join(path_test_npz, 'pred.npy'))
    targets = np.load(os.path.join(path_test_npz, 'target.npy'))
    len_file = len(preds)
    N_minutes_in_one = 30  # 30 predictions are presented in one graph
    # for i in range(len_file // N_minutes_in_one):
    for i in range(10):
        draw(
            preds[i*N_minutes_in_one : (i+1)*N_minutes_in_one, :, :],
            targets[i*N_minutes_in_one : (i+1)*N_minutes_in_one, :, :],
            i,
            path_test_npz
        )

def draw(pred, real, index, save_path):
    len_data = len(pred)
    x_axis = np.arange(24)  # 此处的24表示预测时间长度

    for i in range(len_data):
        pr = pred[i]
        re = real[i]

        pr_sql = pr[:, 0]
        re_sql = re[:, 0]

        # 创建第一个纵轴
        fig, ax1 = plt.subplots()

        # 绘制第一个数据集
        ax1.plot(x_axis, re_sql, 'b', label='real_sql')
        ax1.plot(x_axis, pr_sql, 'b--', label='pred_sql')

        ax1.set_xlabel('IMG:{}'.format(i))
        ax1.set_ylabel('sql_amount', color='b')
        ax1.tick_params('y', colors='b')

        # # 创建第二个纵轴，共享X轴
        # ax2 = ax1.twinx()

        # # 绘制第二个数据集
        # ax2.plot(x_axis, re_sql, 'r', label='real_sql')
        # ax2.plot(x_axis, pr_sql, 'r--', label='pred_sql')
        # ax2.set_ylabel('sql_mount', color='r')
        # ax2.tick_params('y', colors='r')

        # # 添加图例
        # lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # lines = lines1 + lines2
        # labels = labels1 + labels2
        # ax1.legend(lines, labels, loc='upper right')
        plt.savefig(
            os.path.join(
                save_path, str(index) + '_' + str(i) + '.jpg'
            ),
            dpi=300
        )



if __name__ == '__main__':
    main()

