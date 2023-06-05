import numpy as np
import random
import os
import pickle
import torch 
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from uniweat.datasets.utils import create_loader


IDX2LEVEL = {0:50, 1:100, 2:150, 3:200, 4:250, 5:300, 6:400,
             7:500, 8:600, 9:700, 10:850, 11:925, 12:1000}

LEVEL2IDX = {val:key for key, val in IDX2LEVEL.items()}

VAR_DICT = {
    'geopotential':'z', 
    'specific_humidity':'q', 
    'temperature':'t', 
    'u_component_of_wind':'u', 
    'v_component_of_wind':'v', 
    '2m_temperature':'t2m', 
    '10m_u_component_of_wind':'u10', 
    '10m_v_component_of_wind':'v10'
}


class ZTrans(object):
    def __init__(self, 
                 dataset_dir, 
                 var_list, 
                 in_level_list, 
                 out_level_list):
        self.dataset_dir = Path(dataset_dir)
        self.in_level_list = in_level_list
        self.in_level_idx = [LEVEL2IDX[level] for level in in_level_list]
        self.out_level_list = out_level_list
        self.out_level_idx = [LEVEL2IDX[level] for level in out_level_list]

        self.var_list = var_list

        self.const_dict = {'in':{}, 'out':{}}
        for var in var_list:
            pkl_name_mean = self.dataset_dir / var / 'mean.pkl'
            mean_in = self._read_pkl(pkl_name_mean, self.in_level_idx)
            pkl_name_std = self.dataset_dir / var / 'std.pkl'
            std_in = self._read_pkl(pkl_name_std, self.in_level_idx)

            pkl_name_mean = self.dataset_dir / var / 'mean.pkl'
            mean_out = self._read_pkl(pkl_name_mean, self.out_level_idx)
            pkl_name_std = self.dataset_dir / var / 'std.pkl'
            std_out = self._read_pkl(pkl_name_std, self.out_level_idx)

            self.const_dict['in'][var + '_mean'] = mean_in
            self.const_dict['in'][var + '_std'] = std_in
            self.const_dict['out'][var + '_mean'] = mean_out
            self.const_dict['out'][var + '_std'] = std_out

    def _read_pkl(self, pkl_name, level_idx):
        with open(pkl_name, "rb") as pkl_file:
            snap_shot = pickle.load(pkl_file)
            
        if len(snap_shot.shape) == 3:
            snap_shot = snap_shot[level_idx]
        elif len(snap_shot.shape) == 2:
            snap_shot = snap_shot[None, :, :]
        else:
            raise Exception('The file is not standardized pkl. Shape does not match.')

        return snap_shot
    
    def __call__(self, data):
        for io in ['in', 'out']:
            keys = list(data[io].keys())
            for key in keys:
                if key in self.var_list:
                    data[io][key] = (data[io][key] - self.const_dict[io][key + '_mean']) / self.const_dict[io][key + '_std']
        return data


class ConcatTrans(object):
    def __init__(self, 
                 in_concat_keys,
                 out_concat_keys, 
                 in_level_list=None,
                 out_level_list=None):
        self.in_concat_keys = in_concat_keys
        self.out_concat_keys = out_concat_keys

        self.in_level_list = in_level_list
        if in_level_list is not None:
            self.in_level_idx = [LEVEL2IDX[level] for level in in_level_list]
        self.out_level_list = out_level_list
        if out_level_list is not None:
            self.out_level_idx = [LEVEL2IDX[level] for level in out_level_list]

    def __call__(self, data):
        in_list = list()
        for k in self.in_concat_keys:
            in_list.append(data['in'][k])
        out_list = list()
        for k in self.out_concat_keys:
            out_list.append(data['out'][k])
        data_in  = torch.cat(in_list, dim=1)  # [T, C, W, H]
        data_out = torch.cat(out_list, dim=1)

        return data_in, data_out


class MVWeatherBenchDataset(Dataset):
    """Multi-variable Wheather Bench Dataset"""

    def __init__(self,
                 data_root,
                 var_list=[],
                 out_var_list=None,
                 in_level_list=[],
                 out_level_list=[],
                 mode='train',
                 in_time_len=2,
                 out_time_len=12,
                 out_time_res=2,
                 shift_step=1,
                 transforms=None,
                 check_file=True,
                 use_augment=False):
        super().__init__()
        self.data_root = Path(data_root)
        self.time_axis = np.load(Path(data_root)/(mode + '_time_axis.npy'))
        self.space_mesh = torch.from_numpy(
            np.load(Path(data_root)/'space_mesh.npy')
        )
        self.use_augment = use_augment
        self.mean = None
        self.std = None

        self.var_list = var_list
        self.in_level_list = in_level_list
        self.in_level_idx = [LEVEL2IDX[level] for level in in_level_list]
        self.out_level_list = out_level_list
        self.out_level_idx = [LEVEL2IDX[level] for level in out_level_list]
        self.mode = mode
        self.transforms = transforms
        self.check_file(check_file)
        self.snapshot_num = self.time_axis.shape[0]
        self.in_time_len = in_time_len
        self.out_time_len = out_time_len
        if out_var_list is not None:
            self.data_name = out_var_list and len(var_list) >= len(out_var_list)
        else:
            self.data_name = "mv"

        index_for_output = out_time_res*out_time_len

        self.time_windows_list = [np.concatenate((
            self.time_axis[i:i + in_time_len], 
            self.time_axis[i+in_time_len:(i + index_for_output + in_time_len):out_time_res]
            )) for i in range(0, self.snapshot_num - (index_for_output + in_time_len), shift_step)]

    def check_file(self, check_file):
        if check_file:
            print('Checking files...')
            for var in self.var_list:
                target_dir = Path(self.data_root)/var/self.mode
                file_name = ['.'.join(file.split('.')[:-1]) for file in os.listdir(target_dir)]
                file_name = np.sort(np.array(file_name, dtype=np.datetime64))
                if not (file_name == self.time_axis).all():
                    raise Exception('The file is not standardized.')
        else:
            print('Skip checking files.')

    def _read_pkl(self, pkl_name, level_idx):
        with open(pkl_name, "rb") as pkl_file:
            snap_shot = pickle.load(pkl_file)

        if len(snap_shot.shape) == 3:
            snap_shot = snap_shot[level_idx]
        elif len(snap_shot.shape) == 2:
            snap_shot = snap_shot[None, :, :]
        else:
            raise Exception('The file is not standardized pkl. Shape does not match.')

        return snap_shot

    def time_to_float(self, times):
        time_dict = {}
        dates = pd.to_datetime(times)
        time_dict['year'] = torch.tensor(dates.year).float()
        time_dict['month'] = torch.tensor(dates.month).float()
        time_dict['day'] = torch.tensor(dates.day).float()
        time_dict['hour'] = torch.tensor(dates.hour).float()
        # minute, week, second ... 
        return time_dict

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, C, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return len(self.time_windows_list)

    def __getitem__(self, index):
        in_times = self.time_windows_list[index][0:self.in_time_len]
        out_times = self.time_windows_list[index][self.in_time_len:]

        data = {'in':{}, 'out':{}}

        for var in self.var_list:
            
            in_data = []
            for time in in_times:
                pkl_name = self.data_root / var / self.mode / (str(time) + '.pkl')
                snap_shot = self._read_pkl(pkl_name, level_idx=self.in_level_idx)
                in_data.append(torch.from_numpy(snap_shot))
            in_data = torch.stack(in_data, dim=0)

            out_data = []
            for time in out_times:
                pkl_name = self.data_root / var / self.mode / (str(time) + '.pkl')
                snap_shot = self._read_pkl(pkl_name, level_idx=self.out_level_idx)
                out_data.append(torch.from_numpy(snap_shot))
            out_data = torch.stack(out_data, dim=0)

            if self.use_augment:
                data['in'][var] = self._augment_seq(in_data, crop_scale=0.96)
                data['out'][var] = self._augment_seq(out_data, crop_scale=0.96)
            else:
                data['in'][var] = in_data
                data['out'][var] = out_data

        data['in']['time'] = self.time_to_float(in_times)
        data['out']['time'] = self.time_to_float(out_times)
        data['in']['space'] = self.space_mesh
        data['out']['space'] = self.space_mesh

        if self.transforms is not None:
            data = self.transforms(data)

        # return data, labels
        return data


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              data_split='1_40625',
              in_var_list=[v for q, v in VAR_DICT.items()],
              out_var_list=['z'],
              in_level_list=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
              out_level_list=[500],
              in_time_len=4,
              out_time_len=4,
              out_time_res=2,
              shift_step=6,
              transforms=None,
              check_file=True,
              distributed=False, use_augment=False, use_prefetcher=False,
              **kwargs):

    assert data_split in ['5_625', '2_8125', '1_40625']
    _dataroot = os.path.join(data_root, f'mv_weather_{data_split}deg')
    weather_dataroot = _dataroot if os.path.exists(_dataroot) else '/linhaitao/standard_pkl_811'
    print('check data_root:', weather_dataroot)

    if transforms is None:
        ztrans = ZTrans(
            weather_dataroot, in_var_list, in_level_list=in_level_list, out_level_list=out_level_list)
        ctrans = ConcatTrans(
            in_concat_keys=in_var_list, out_concat_keys=out_var_list)
        transforms = Compose([ztrans, ctrans])
    else:
        transforms = Compose(transforms) if isinstance(transforms, list) else None

    train_set = MVWeatherBenchDataset(
        data_root=weather_dataroot,
        var_list=in_var_list, out_var_list=out_var_list,
        in_level_list=in_level_list, out_level_list=out_level_list,
        mode='train', check_file=check_file, in_time_len=in_time_len,
        out_time_len=out_time_len, out_time_res=out_time_res,
        shift_step=shift_step, transforms=transforms,
        use_augment=use_augment)
    vali_set = MVWeatherBenchDataset(
        data_root=weather_dataroot,
        var_list=in_var_list, out_var_list=out_var_list,
        in_level_list=in_level_list, out_level_list=out_level_list,
        mode='val', check_file=check_file, in_time_len=in_time_len,
        out_time_len=out_time_len, out_time_res=out_time_res,
        shift_step=shift_step, transforms=transforms,
        use_augment=use_augment)
    test_set = MVWeatherBenchDataset(
        data_root=weather_dataroot,
        var_list=in_var_list, out_var_list=out_var_list,
        in_level_list=in_level_list, out_level_list=out_level_list,
        mode='test', check_file=check_file, in_time_len=in_time_len,
        out_time_len=out_time_len, out_time_res=out_time_res,
        shift_step=shift_step, transforms=transforms,
        use_augment=use_augment)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(vali_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    import time

    data_root = '/linhaitao/standard_pkl_811'
    data_split = '1_40625'
    in_var_list = ['z', 'q', 't', 'u', 'v', 't2m', 'u10', 'v10']
    out_var_list = ['z', 'q', 't', 'u', 'v']
    # in_level_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    in_level_list = [100, 300, 500, 700, 850, 925, 1000]
    out_level_list = [500]

    global_start = time.time()
    dataloader_train, _, dataloader_test = \
        load_data(
            batch_size=16, val_batch_size=16,
            data_root=data_root, num_workers=8,
            in_var_list=in_var_list,
            out_var_list=out_var_list,
            in_level_list=in_level_list,
            out_level_list=out_level_list,
            in_time_len=4,
            out_time_len=4,
            out_time_res=2,
            shift_step=6,
            transforms=None, check_file=False,
            distributed=False, use_augment=False, use_prefetcher=False,
        )
    print(len(dataloader_train), len(dataloader_test))

    start = time.time()
    for i, item in enumerate(dataloader_train):
        print(i, item[0].shape, item[1].shape, "train data time={}".format(time.time() - start))
        start = time.time()
        if i > 16: break
    start = time.time()
    for i, item in enumerate(dataloader_test):
        print(i, item[0].shape, item[1].shape, "test data time={}".format(time.time() - start))
        start = time.time()
        if i > 16: break
