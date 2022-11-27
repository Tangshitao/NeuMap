import numpy as np


def x_2d_coords(h, w):
    x_2d = np.zeros((h, w, 2), dtype=np.float32)
    for y in range(0, h):
        x_2d[y, :, 1] = y
    for x in range(0, w):
        x_2d[:, x, 0] = x
    return x_2d

def pi_inv(K, x, d):
    fx, fy, cx, cy = K[0:1, 0:1], K[1:2, 1:2], K[0:1, 2:3], K[1:2, 2:3]
    X_x = d * (x[:, :, 0] - cx) / fx
    X_y = d * (x[:, :, 1] - cy) / fy
    X_z = d
    X = np.stack([X_x, X_y, X_z], axis=0).transpose([1, 2, 0])
    return X

def inv_pose(R, t):
    Rwc = R.T
    tw = -Rwc.dot(t)
    return Rwc, tw

def transpose(R, t, X):
    X = X.reshape(-1, 3)
    X_after_R = R.dot(X.T).T
    trans_X = X_after_R + t
    return trans_X

def back_projection(depth, pose, K):
    h, w = depth.shape
    x_2d = x_2d_coords(h, w)

    X_3d = pi_inv(K, x_2d, depth)
    Rwc, twc = pose[:3, :3], pose[:3, 3]
    X_world = transpose(Rwc, twc, X_3d)

    X_world = X_world.reshape((h, w, 3))
    return X_world, x_2d