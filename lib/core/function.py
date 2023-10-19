# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by BgZhang
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, os
import logging
import torch
import numpy as np
from tqdm import tqdm
# from core.evaluate import accuracy
from core.metrics import metric
# from sklearn import metrics
# from sklearn.metrics import roc_auc_score, auc
# from trick_dl.freeze import freeze_model, freeze_bn, activate_bn


logger = logging.getLogger(__name__)


def process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, config, model):
    batch_x = batch_x.float().cuda()
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().cuda()
    batch_y_mark = batch_y_mark.float().cuda()

    # decoder input
    if config["MODEL"]["PADDING"] == 0:
        dec_inp = torch.zeros(
            [batch_y.shape[0], config["DATASET"]["PRED_LEN"], batch_y.shape[-1]]).float()  # torch.Size([32, 24, 7])
    elif config["MODEL"]["PADDING"] == 1:
        dec_inp = torch.ones(
            [batch_y.shape[0], config["DATASET"]["PRED_LEN"], batch_y.shape[-1]]).float()

    dec_inp = torch.cat(
        [batch_y[:, :config["DATASET"]["LABEL_LEN"], :], dec_inp], dim=1).float().cuda()  # torch.Size([32, 72, 7])
    # encoder - decoder
    # if self.args.use_amp:
    #     with torch.cuda.amp.autocast():
    #         if self.args.output_attention:
    #             outputs = self.model(
    #                 batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #         else:
    #             outputs = self.model(
    #                 batch_x, batch_x_mark, dec_inp, batch_y_mark)
    # else:
    if config["MODEL"]["OUTPUT_ATTENTION"]:
        outputs = model(batch_x, batch_x_mark,
                        dec_inp, batch_y_mark)[0]
    else:
        outputs = model(batch_x, batch_x_mark,
                        dec_inp, batch_y_mark)  # torch.Size([32, 24, 7])
    # f_dim = -1 if self.args.features == 'MS' else 0  # 0
    # batch_y = batch_y[:, -self.args.pred_len:,
    #                     f_dim:].to(self.device)  # torch.Size([32, 24, 7])
    f_dim = 0
    batch_y = batch_y[:, -config["DATASET"]["PRED_LEN"]:, f_dim:].cuda()  # torch.Size([32, 24, 7])
    return outputs, batch_y


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        # input = input.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        pred, target = process_one_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark, config, model)

        loss = config['TRAIN']['LOSS_WEIGHT'] * criterion(pred, target)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    msg = 'Epoch: [{0}]\t' \
        'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
        'Speed {speed:.1f} samples/s\t' \
        'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
        'LR {learning_rate}\t' \
        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
            epoch, batch_time=batch_time,
            speed=target.size(0)/batch_time.val,
            data_time=data_time,
            learning_rate=optimizer.state_dict()['param_groups'][0]['lr'],
            loss=losses)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer.add_scalar('train_lr', optimizer.state_dict()[
                          'param_groups'][0]['lr'], global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None, prefix='valid'):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        end = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(val_loader)):
            # compute output
            pred, target = process_one_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark, config, model)
            preds.append(pred.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        preds = np.array(preds)
        targets = np.array(targets)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        targets = targets.reshape(-1, targets.shape[-2], targets.shape[-1])
        mae, mse, rmse, mape, mspe = metric(preds, targets)

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
            'MAE Loss {mae:.4f}\t' \
            'MSE Loss {mse:.4f}\t' \
            'RMSE Loss {rmse:.4f}\t' \
            'MAPE Loss {mape:.4f}\t' \
            'MSPE Loss {mspe:.4f}'.format(
                batch_time=batch_time, mae=mae, mse=mse, rmse=rmse, mape=mape, mspe=mspe)

    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict[f'{prefix}_global_steps']
        writer.add_scalar(f'{prefix}_mae', mae, global_steps)
        writer.add_scalar(f'{prefix}_mse', mse, global_steps)
        writer.add_scalar(f'{prefix}_rmse', rmse, global_steps)
        writer.add_scalar(f'{prefix}_mape', mape, global_steps)
        writer.add_scalar(f'{prefix}_mspe', mspe, global_steps)
        writer_dict[f'{prefix}_global_steps'] = global_steps + 1

    return mse


def draw_validate(config, valid_loader, model, final_output_dir):
    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(valid_loader)):
            # compute output
            pred, target = process_one_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark, config, model)
            preds.append(pred.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())

        preds = np.array(preds)
        targets = np.array(targets)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        targets = targets.reshape(-1, targets.shape[-2], targets.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, targets)
        msg = 'mse:{}, mae:{}'.format(mse, mae)
        logger.info(msg)

        np.save(
            os.path.join(final_output_dir, 'metrics.npy'),
            np.array([mae, mse, rmse, mape, mspe])
        )
        np.save(
            os.path.join(final_output_dir, 'pred.npy'),
            preds
        )
        np.save(
            os.path.join(final_output_dir, 'target.npy'),
            targets
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # 表示正确率
        self.sum += val * n  # 表示正确的个数
        self.count += n
        self.avg = self.sum / self.count
