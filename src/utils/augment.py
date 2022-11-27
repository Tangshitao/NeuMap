from imgaug import augmenters as iaa
import random
import numpy as np


class _AffineMatrixGenerator(object):
    # Added in 0.5.0.
    def __init__(self, matrix=None):
        if matrix is None:
            matrix = np.eye(3, dtype=np.float32)
        self.matrix = matrix

    # Added in 0.5.0.
    def centerize(self, image_shape):
        height, width = image_shape[0:2]
        self.translate(-width/2, -height/2)
        return self

    # Added in 0.5.0.
    def invert_centerize(self, image_shape):
        height, width = image_shape[0:2]
        self.translate(width/2, height/2)
        return self

    # Added in 0.5.0.
    def translate(self, x_px, y_px):
        if x_px < 1e-4 or x_px > 1e-4 or y_px < 1e-4 or x_px > 1e-4:
            matrix = np.array([
                [1, 0, x_px],
                [0, 1, y_px],
                [0, 0, 1]
            ], dtype=np.float32)
            self._mul(matrix)
        return self

    # Added in 0.5.0.
    def scale(self, x_frac, y_frac):
        if (x_frac < 1.0-1e-4 or x_frac > 1.0+1e-4
                or y_frac < 1.0-1e-4 or y_frac > 1.0+1e-4):
            matrix = np.array([
                [x_frac, 0, 0],
                [0, y_frac, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            self._mul(matrix)
        return self

    # Added in 0.5.0.
    def rotate(self, rad):
        if rad < 1e-4 or rad > 1e-4:
            rad = -rad
            matrix = np.array([
                [np.cos(rad), np.sin(rad), 0],
                [-np.sin(rad), np.cos(rad), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            self._mul(matrix)
        return self

    # Added in 0.5.0.
    def shear(self, x_rad, y_rad):
        if x_rad < 1e-4 or x_rad > 1e-4 or y_rad < 1e-4 or y_rad > 1e-4:
            matrix = np.array([
                [1, np.tanh(-x_rad), 0],
                [np.tanh(y_rad), 1, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            self._mul(matrix)
        return self

    # Added in 0.5.0.
    def _mul(self, matrix):
        self.matrix = np.matmul(matrix, self.matrix)

def data_aug_dense(img, coord, xy): # coord shape (H, W, 3)
    trans_x =random.uniform(-0.2,0.2)
    trans_y =random.uniform(-0.2,0.2)

    aug_add = iaa.Add(random.randint(-20,20))

    scale=random.uniform(0.7,1.5)
    rotate=random.uniform(-30,30)
    shear=random.uniform(-10,10)

    aug_affine = iaa.Affine(scale=scale,rotate=rotate,
                shear=shear,translate_percent={"x": trans_x, "y": trans_y}) 
    aug_affine_coords = iaa.Affine(scale=scale,rotate=rotate,
                    shear=shear,translate_percent={"x": trans_x, "y": trans_y},
                    order=0) 
    img = aug_add.augment_image(img) 
    img = aug_affine.augment_image(img)
    input_concat=np.concatenate([coord[:,:,None], xy], axis=-1)
    input_concat = aug_affine_coords.augment_image(input_concat)
    xy=input_concat[:,:,1:]
    coord=input_concat[:,:,0]
   
    return img, coord, xy

def data_aug(img, coord, xy, range_mask, img_h, img_w): # coord shape (N, 3)
    trans_x =random.uniform(-0.2,0.2)
    trans_y =random.uniform(-0.2,0.2)

    aug_add = iaa.Add(random.randint(-20,20))

    scale=random.uniform(0.7,1.5)
    rotate=random.uniform(-30,30)
    shear=random.uniform(-10,10)

    aug_affine = iaa.Affine(scale=scale,rotate=rotate,
                shear=shear,translate_percent={"x": trans_x, "y": trans_y}) 
  
    img = aug_add.augment_image(img) 
    img = aug_affine.augment_image(img)

    xy[:, 0]-=img_w/2
    xy[:, 1]-=img_h/2
    aff_mtx=_AffineMatrixGenerator()
    aff_mtx.scale(scale, scale)
    aff_mtx.shear(shear/180*np.pi, 0)
    aff_mtx.rotate(rotate/180*np.pi)
    
    aff_mtx.translate(trans_x*img_w, trans_y*img_h)
    mtx=aff_mtx.matrix
    N=xy.shape[0]
    xyone=np.concatenate([xy.transpose((1, 0)), np.ones((1, N))])
    xyone=mtx@xyone
    xy_trans=(xyone[:2]/xyone[2:]).transpose((1, 0))
    xy_trans[:,0]+=img_w/2
    xy_trans[:,1]+=img_h/2
    mask=(xy_trans[:,0]>=0)*(xy_trans[:,0]<img_w)*(xy_trans[:,1]>=0)*(xy_trans[:,1]<img_h)*range_mask

    return img, coord, xy_trans, mask
