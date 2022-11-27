import imp
from os import path as osp
from typing import Dict
from unicodedata import name
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils as utils
import cv2
from numpy.linalg import inv
from src.utils.dataset import (
    get_depth,
    get_coord
)
from src.utils.augment import data_aug_dense
from src.utils.geometry import *
from skimage import io, color
from torchvision import transforms
from src.utils.dataset import pad_matrix

class ScanNetDataset(utils.data.Dataset):
    def __init__(self,
                 root,
                 train_path,
                 mode='train',
                 image_size=480,
                 max_n_points=20000,
                 rgb=False,
                 **kwargs):
        super().__init__()
        self.image_size=image_size
        self.max_n_points=max_n_points
        
        self.rgb=rgb
        self.root = os.path.join(root, 'train')
        self.mode = mode

        if self.mode=='train':
            clr_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=[0,0])
        else:
            clr_jitter = transforms.ColorJitter(saturation=[0,0])
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, int(image_size/3*4))),
            clr_jitter,
            transforms.ToTensor()
            ])
        
        self.train_list=[]
        
        if self.mode=='train':
            self.voxel_id=train_path
            info=np.load(os.path.join(root, train_path), allow_pickle=True).item()
            self.idx_list=info['image_names']           
        else:
            self.idx_list=[]
            
            query_voxel_id_map=np.load(os.path.join(root, train_path),allow_pickle=True).item()
           
            self.idx_list=[]
            self.voxel_id=[]
            
            for k, v in query_voxel_id_map.items():
                self.idx_list.append(k)
                self.voxel_id.append(list(v))

    def __len__(self):
        return len(self.idx_list)

    def _read_abs_pose(self, scene_name, name):
        pth = os.path.join(self.root,
                       scene_name,
                       'pose', f'{name}.txt')
        return np.loadtxt(pth)
    
    def get_coords(self, depth, xy, color_K, gt_pose):
        depth = depth[4::8,4::8]
        xy=xy[4::8, 4::8]
        X_3d = pi_inv(color_K, xy, depth)
        
        Rwc, twc = gt_pose[:3, :3], gt_pose[:3, 3]
        coords = transpose(Rwc, twc, X_3d)
        mask=(depth>0).reshape(-1)

        return coords.reshape(-1, 3), mask.reshape(-1), xy.reshape(-1, 2)
    
    def __getitem__(self, index):
        scene, image_id = self.idx_list[index]
        image_id=image_id[:-4]

        if self.rgb:
            with open(os.path.join(self.root, scene,'color', image_id+'.jpg'), 'rb') as f:
                check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9': # broken image
                return self.__getitem__(random.randint(0, self.__len__()-1))
            image = cv2.imread(os.path.join(self.root, scene,'color', image_id+'.jpg'))
            image_h, image_w, _=image.shape
        else:
            image = cv2.imread(os.path.join(self.root, scene,'color', image_id+'.jpg'), cv2.IMREAD_GRAYSCALE)
            image_h, image_w=image.shape
        scale_factor = self.image_size / min(image_h, image_w)
        
        # camera parameters
        K=np.loadtxt(os.path.join(self.root, scene, 'intrinsic/intrinsic_color.txt'))[:3,:3]
        camera_params=np.array([K[0,0], K[1, 1], K[0, 2], K[1, 2]])
        camera_type='PINHOLE'

        K=K*scale_factor
        K[-1,-1]=1
       
        gt_pose=self._read_abs_pose(scene, image_id)
        depth=cv2.imread(os.path.join(self.root, scene, 'depth', image_id+'.png'),-1)/1000
        h, w=depth.shape
        xy=x_2d_coords(h, w)
        
        if self.mode=='train':
            image, depth, xy_aug=data_aug_dense(image, depth, xy)

            image=self.image_transform(image)
            voxel_id=self.voxel_id
            gt_coords, mask, _=self.get_coords(depth, xy_aug, K, gt_pose)
            xy=xy[4::8, 4::8].reshape(-1, 2)

            gt_coords=pad_matrix(gt_coords[mask], self.max_n_points)
            xy=pad_matrix(xy[mask], self.max_n_points)
            mask=np.ones_like(mask).astype(np.bool)
        else:
            xy_aug=xy
            image=self.image_transform(image)
            C, H, W=image.shape
            new_H=((H-1)//8+1)*8
            new_W=((W-1)//8+1)*8
            
            new_image=torch.zeros((C, new_H, new_W)).float()
            new_image[:, :H,:W]=image[:, :H,:W]
            image=new_image
            voxel_id=self.voxel_id[index]
        
            gt_coords=np.zeros((new_H*new_W//64, 3)).astype(np.float32)
            mask=np.ones((new_H*new_W//64)).astype(np.bool)
            xy=xy[4::8, 4::8].reshape(-1, 2)

        if self.mode=='test':
            image_name=image_id.split('/')[-1]                
        else:
            image_name='scannet'
       
        data = {
            'image_h':image_h, 
            'image_w':image_w,
            'xy': xy,
            'mask':mask,
            'image': image,   # (1, h, w)
            'gt_pose' : gt_pose,
            'camera_params': camera_params,  # (3, 3)
            'camera_type':camera_type,
            'gt_coords':gt_coords,
            'dataset_name': 'scannet',
            'scale_factor':scale_factor,
            'image_name': image_name,
            'mask': mask,
            'voxel_id':voxel_id,
        }
        return data