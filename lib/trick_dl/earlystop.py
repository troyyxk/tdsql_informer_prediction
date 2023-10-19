import numpy as np
import torch
import logging

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, optimizer, patience=1, delta=0, min_lr=1e-6, flag=0):
        """
        Args:
            optimizer (优化器): 优化器
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            min_lr (float): 学习率最小值,小于该最小值时,停止训练
            flag (int): 表示传入的score值是越大越好(0),还是越小越好(1). 即对于图片识别准确率,此处为0, 对于图片相似性,则为1
        """
        self.optimizer = optimizer
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.score_max = -np.Inf
        self.delta = delta
        self.min_lr = min_lr
        self.flag = flag

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.flag == 0 and score < self.best_score - self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.flag == 1 and score > self.best_score - self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def adjust_learning_rate(self):
        self.early_stop = False
        self.counter = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < self.min_lr:
                return False
        return True