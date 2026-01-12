import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob
from pathlib import Path

from .augment import Augmentor

import imageio.v2 as imageio

class DSECfull(data.Dataset):
    def __init__(self, phase):
        self.init_seed = False
        self.phase = phase
        self.files = []
        self.flows = []

        assert self.phase in ["train", "trainval", "test", "prog"]

        crop_size = [288, 384]
        flip = True

        ### Please change the root to satisfy your data saving setting.
        root = 'C:/users/abdessamad/TMA/datasets/dsec_full'
        if phase == 'train' or phase == 'trainval':
            self.root = os.path.join(root, 'trainval')
            self.augmentor = Augmentor(crop_size, do_flip=flip)
        else:
            self.root = os.path.join(root, 'test')

        if phase == 'prog':
            self.augment = False
            self.root = 'datasets/dsec_prog/trainval'

        self.files = glob.glob(os.path.join(self.root, '*', 'voxel_grids', '*.npz'))
        self.files.sort()

        self.depths = glob.glob(os.path.join(self.root, '*', 'depth_gt', "*.png"))
        self.depths.sort()

        self.images = glob.glob(os.path.join(self.root, '*', 'images_original', "*.png"))
        self.depths.sort()

    def events_to_voxel_grid(self, x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
            }
        return self.representation.convert(event_data_torch)

    
    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        #events
        events_file = np.load(self.files[index])
        event_voxel = events_file['events_curr'].transpose(1, 2, 0)

        img = imageio.imread(self.images[index])

        if self.phase != 'test':
            depth_16bit = imageio.imread(self.depths[index], format="PNG-FI")
            depth = depth_16bit.astype(np.float32) / 256.0
            valid2D = (depth > 0).astype(int)

            if self.phase == 'train':
                event_voxel, depth, valid2D, img= self.augmentor(event_voxel, depth, valid2D, img)

            img = torch.from_numpy(img).permute(2, 0, 1).float()
            depth = torch.from_numpy(depth).unsqueeze(0).float()
            valid2D = torch.from_numpy(valid2D).float()

        if self.phase == 'test':
            # Include submission coordinates (seuence name, file index)
            file_path = Path(self.files[index])
            sequence_name = file_path.parents[1].name
            file_index = int(file_path.stem)
            submission_coords = (sequence_name, file_index)
            return event_voxel, img, submission_coords
    
        event_voxel = torch.from_numpy(event_voxel).permute(2, 0, 1).float()
        return event_voxel, depth, valid2D, img
    
    def __len__(self):
        return len(self.files)


def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D


def make_data_loader(phase, batch_size, num_workers):
    dset = DSECfull(phase)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True)
    return loader

if __name__ == '__main__':
    dset = DSECfull('train')
    print(len(dset))
    voxel, depth, valid, img = dset[0]
    print(voxel.shape, depth.shape, valid.shape, img.shape)