from torchvision import transforms
import os
import random
import numpy as np
from torch.utils import data
import pickle as pkl
import cv2
import torch
import quaternion
import random
from src.utils.augment import data_aug
from src.utils.dataset import pad_matrix


class KaptureDataset(data.Dataset):
    def __init__(self, root, kapture_data, sensor_dict, train_path, input_path, mode='train',image_size=640, max_n_points=20000, random_crop=True, rgb=False, aspect_ratio=None):
        self.max_n_points=max_n_points
        self.train_path=train_path
        self.root=root
        self.mode=mode
        self.image_size=image_size
        self.input_path=input_path
        self.random_crop=random_crop
        self.rgb=rgb
        self.aspect_ratio=aspect_ratio
        if self.mode=='train':
            clr_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=[0,0])
        else:
            clr_jitter = transforms.ColorJitter(saturation=[0,0])
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            clr_jitter,
            transforms.ToTensor()
            ])
        self.sensor_dict=sensor_dict
        self.kaptures=kapture_data

        self.load_meta()
    
    def load_meta(self):
        if self.mode=='train':
            self.voxel_id=self.train_path
            info=np.load(os.path.join(self.root, self.train_path), allow_pickle=True).item()
            self.idx_list=info['image_names']           
        else:
            self.idx_list=[]
            query_voxel_id_map=np.load(os.path.join(self.root, self.train_path),allow_pickle=True).item()
            self.idx_list=[]
            self.voxel_id=[]
            
            for k, v in query_voxel_id_map.items():
                self.idx_list.append(k)
                self.voxel_id.append(list(v))

    def __len__(self):
        return len(self.idx_list)
    
    def load_pose(self, timestep, sensor_id):
        
        if self.kaptures.trajectories is not None and (timestep, sensor_id) in self.kaptures.trajectories:
            pose_world_to_cam = self.kaptures.trajectories[(timestep, sensor_id)]
            pose_world_to_cam_matrix = np.zeros((4, 4), dtype=np.float)
            pose_world_to_cam_matrix[0:3, 0:3] = quaternion.as_rotation_matrix(pose_world_to_cam.r)
            pose_world_to_cam_matrix[0:3, 3] = pose_world_to_cam.t_raw
            pose_world_to_cam_matrix[3, 3] = 1.0
            T = torch.tensor(pose_world_to_cam_matrix).float()
            gt_pose=T.inverse()
        else:
            gt_pose=T=torch.eye(4)
        return gt_pose
    
    def crop_image(self, image, xy, mask):
        C, H, W=image.shape
        image_crop=torch.zeros((C, self.image_size, self.image_size)).float()
        margin_H=H-self.image_size
        margin_W=W-self.image_size
        
        if H>self.image_size:
            crop_size_H=random.randint(0, margin_H-1)
            crop_image_H=self.image_size
        else:
            crop_size_H=0
            crop_image_H=H
        if W>self.image_size:
            crop_size_W=random.randint(0, margin_W-1)
            crop_image_W=self.image_size
        else:
            crop_size_W=0
            crop_image_W=W
        
        image_crop[:, :crop_image_H:,:crop_image_W]=image[:, crop_size_H:crop_size_H+crop_image_H:,crop_size_W:crop_size_W+crop_image_W]
        xy[:,0]=xy[:,0]-crop_size_W
        xy[:,1]=xy[:,1]-crop_size_H
        mask=mask*(xy[:,1]>=0)*(xy[:,1]<crop_image_W)*(xy[:,0]>=0)*(xy[:,0]<crop_image_H)
        return image_crop, xy, mask

    def __getitem__(self, index):
        image_id = self.idx_list[index]
        timestep, sensor_id=self.sensor_dict[image_id]

        points_path=os.path.join(os.path.join(self.input_path, 'points', image_id+'.npy'))
        if not os.path.exists(points_path):
            return self.__getitem__(random.randint(0, len(self.idx_list)-1))
        
        # load image
        if self.rgb:
            image = cv2.imread(os.path.join(self.input_path, 'sensors/records_data', image_id))
            image_h, image_w, _=image.shape
        else:
            image = cv2.imread(os.path.join(self.input_path, 'sensors/records_data', image_id), cv2.IMREAD_GRAYSCALE)
            image_h, image_w=image.shape
        
        scale_factor = self.image_size / min(image_h, image_w)
        
        # load camera parameters
        camera_params=np.array(self.kaptures.sensors[sensor_id].camera_params[2:])
        camera_type=str(self.kaptures.sensors[sensor_id].camera_type).split('.')[-1]
        gt_pose=self.load_pose(timestep, sensor_id)

        # load points
        points=np.load(points_path, allow_pickle=True).item()
        xy=points['xy']

        if self.mode=='train':
            gt_coords=points['points'][:,:3].astype(np.float32)
            mask=points['mask']

            # random rotation, scale, shift
            image, gt_coords, xy, mask=data_aug(image, gt_coords, xy, mask, image_h, image_w)
            
            # color jitter
            image=self.image_transform(image)
            
            # rescale xy 
            xy=(xy*scale_factor).astype(np.float32)

            # random crop image
            if self.random_crop:
                image, xy, mask=self.crop_image(image, xy, mask) 
            elif self.aspect_ratio is not None:
                c, _image_h, _image_w=image.shape
                new_image_w=int(_image_h/self.aspect_ratio)
                if new_image_w!=_image_w:
                    new_image=torch.zeros(c, _image_h, new_image_w).float()
                    new_image[:,:,:_image_w]=image
                    image=new_image

            voxel_id=self.voxel_id
            
            # padding to the same dimensions
            xy=pad_matrix(xy, self.max_n_points)
            gt_coords=pad_matrix(gt_coords, self.max_n_points)
            mask=pad_matrix(mask[:,None], self.max_n_points)[:,0].astype(np.bool)
        else:
            # dummy gt coords
            gt_coords=np.zeros((xy.shape[0],3), dtype=np.float32)
            mask=np.zeros(xy.shape[0], dtype=np.bool)

            # resize image, make sure H and W can be divided by 8
            image=self.image_transform(image)
            C, H, W=image.shape
            new_H=((H-1)//8+1)*8
            new_W=((W-1)//8+1)*8
            new_image=torch.zeros((C, new_H, new_W)).float()
            new_image[:, :H,:W]=image[:, :H,:W]
            image=new_image

            voxel_id=self.voxel_id[index]
        
            xy=(points['xy']*scale_factor).astype(np.float32)

        if self.mode=='test':
            image_name=image_id.split('/')[-1]                
        else:
            image_name='kapture'

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
            'dataset_name': 'kapture',
            'scale_factor':scale_factor,
            'image_name': image_name,
            'mask': mask,
            'voxel_id':voxel_id,
        }
        return data