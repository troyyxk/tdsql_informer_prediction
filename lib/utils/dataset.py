import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
from .timefeatures import time_features

class Dataset_TDSQL(Dataset):
    def __init__(self,
                 file_path:str,
                 con_name_select: list,
                 flag='train',
                 inverse=False,
                 timeenc=1,
                 freq='T',
                 seq_len=96,
                 label_len=48,
                 pred_len=24
                 ):
        """TDSQL DATASET

        Args:
            file_path (str): dataset dir path. There will be *.csv in this dir.
            con_name_select (list): the col for select
            flag (str, optional): 'train', 'test' or 'val'. Defaults to 'train'.
            inverse (bool, optional): will be deleted. Defaults to False.
            timeenc (int, optional): reference to timefeature.py. Defaults to 1.
            freq (str, optional): reference to timefeature.py. Defaults to 'T'.
            seq_len (int, optional): reference to doc/TDSQLAI.pptx. Defaults to 96.
            label_len (int, optional): reference to doc/TDSQLAI.pptx. Defaults to 48.
            pred_len (int, optional): reference to doc/TDSQLAI.pptx. Defaults to 24.
        """

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.con_name_select = con_name_select


        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.file_path = file_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.file_path)
        len_df_raw = len(df_raw)

        border1s = [
            0,
            len_df_raw - 60 * 24 * 7 * 2 - self.seq_len,
            len_df_raw - self.seq_len - 60 * 24 * 7 * 1
        ]
        border2s = [
            len_df_raw - 60 * 24 * 7 * 2,
            len_df_raw - 60 * 24 * 7 * 1,
            len_df_raw - 1
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # TODO: 这里需要修改,我根据数据集,写成了32
        # cols_data = df_raw.columns[1:]
        # cols_data = df_raw.columns[1:33]
        # df_data = df_raw[cols_data]

        # df_data = df_raw[self.con_name_select]
        df_data = df_raw[self.con_name_select].apply(lambda x: x.sum(), axis=1)
        # df_data = df_data.div(16.0)
        data = df_data.values.reshape(-1, 1)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
