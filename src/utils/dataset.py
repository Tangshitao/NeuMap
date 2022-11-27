import io
from loguru import logger

import numpy as np
import h5py
import torch
from numpy.linalg import inv


try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None

# --- DATA IO ---

def pad_matrix(matrix, n_points):
    n, d=matrix.shape
    new_matrix=np.zeros((n_points, d)).astype(np.float32)
   
    idx=0 
    for i in range(n_points//n):
        new_matrix[idx:idx+n]=matrix
        idx+=n

    n_points-=idx
    new_matrix[idx:]=matrix[:n_points]
    return new_matrix


def get_depth(depth, calibration_extrinsics, intrinsics_color,
              intrinsics_depth_inv):
    """Return the calibrated depth image (7-Scenes). 
    Calibration parameters from DSAC (https://github.com/cvlab-dresden/DSAC) 
    are used.
    """
    img_height, img_width = depth.shape[0], depth.shape[1]
    depth_ = np.zeros_like(depth)
    x = np.linspace(0, img_width-1, img_width)
    y = np.linspace(0, img_height-1, img_height)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, (1, -1))
    yy = np.reshape(yy, (1, -1))
    ones = np.ones_like(xx)
    pcoord_depth = np.concatenate((xx, yy, ones), axis=0)
    depth = np.reshape(depth, (1, img_height*img_width))
    ccoord_depth = np.dot(intrinsics_depth_inv, pcoord_depth) * depth
    ccoord_depth[1,:] = - ccoord_depth[1,:]
    ccoord_depth[2,:] = - ccoord_depth[2,:]
    ccoord_depth = np.concatenate((ccoord_depth, ones), axis=0)
    ccoord_color = np.dot(calibration_extrinsics, ccoord_depth)
    ccoord_color = ccoord_color[0:3,:]
    ccoord_color[1,:] = - ccoord_color[1,:]
    ccoord_color[2,:] = depth

    pcoord_color = np.dot(intrinsics_color, ccoord_color)
    pcoord_color = pcoord_color[:,pcoord_color[2,:]!=0]
    pcoord_color[0,:] = pcoord_color[0,:]/pcoord_color[2,:]+0.5
    pcoord_color[0,:] = pcoord_color[0,:].astype(int)
    pcoord_color[1,:] = pcoord_color[1,:]/pcoord_color[2,:]+0.5
    pcoord_color[1,:] = pcoord_color[1,:].astype(int)
    pcoord_color = pcoord_color[:,pcoord_color[0,:]>=0]
    pcoord_color = pcoord_color[:,pcoord_color[1,:]>=0]
    pcoord_color = pcoord_color[:,pcoord_color[0,:]<img_width]
    pcoord_color = pcoord_color[:,pcoord_color[1,:]<img_height]

    depth_[pcoord_color[1,:].astype(int),
           pcoord_color[0,:].astype(int)] = pcoord_color[2,:]
    return depth_

def get_coord(depth, pose, intrinsics_color_inv):
    """Generate the ground truth scene coordinates from depth and pose.
    """
    img_height, img_width = depth.shape[0], depth.shape[1]
    mask = np.ones_like(depth)
    mask[depth==0] = 0
    mask = np.reshape(mask, (img_height, img_width,1))
    x = np.linspace(0, img_width-1, img_width)
    y = np.linspace(0, img_height-1, img_height)
    xx, yy = np.meshgrid(x, y)
    xy=np.stack([xx, yy], axis=-1)
    
    xx = np.reshape(xx, (1, -1))
    yy = np.reshape(yy, (1, -1))
    ones = np.ones_like(xx)
    
    pcoord = np.concatenate((xx, yy, ones), axis=0)
    depth = np.reshape(depth, (1, img_height*img_width))
    ccoord = np.dot(intrinsics_color_inv, pcoord) * depth
    ccoord = np.concatenate((ccoord, ones), axis=0)
    scoord = np.dot(pose, ccoord)
    scoord = np.swapaxes(scoord,0,1)
    scoord = scoord[:,0:3]
    scoord = np.reshape(scoord, (img_height, img_width,3))
    scoord = scoord * mask
    mask = np.reshape(mask, (img_height, img_width))
    return scoord, mask, xy