import torch
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
import pycolmap
import numpy as np
import math

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

# --- METRICS ---
def rel_rot_quaternion_deg(q1, q2):
    """
    Compute relative error (deg) of two quaternion
    :param q1: quaternion 1, (w, x, y, z), dim: (4)
    :param q2: quaternion 2, (w, x, y, z), dim: (4)
    :return: relative angle in deg
    """
    return 2 * 180 * np.arccos(np.clip(np.dot(q1, q2), -1.0, 1.0)) / np.pi


def rel_rot_angle(T1, T2):
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    q1 = quaternion_from_matrix(R1)
    q2 = quaternion_from_matrix(R2)
    return rel_rot_quaternion_deg(q1, q2)

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)+1e-6
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    t_err_abs=np.linalg.norm(t-t_gt)
    r_err_abs=rel_rot_angle(T_0to1[:3,:3],R)
    r_err_abs=np.minimum(r_err_abs, 360-r_err_abs)
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err, t_err_abs, r_err_abs

def compute_pose(coords_list, scores_list, xy_list, scale_factor, score_thresh, bs, image_h, image_w, camera_type, camera_params, ransac_thresh):
    _xy_list=[]
    _coords_list=[]
    ret, ret_dict=None, None
    for coords, scores, xy in zip(coords_list, scores_list, xy_list):
        mask=scores[bs]>score_thresh
        if mask.sum()>40:
            _coords=coords[bs, mask].cpu().numpy()
            _xy=(xy[bs, mask]/scale_factor).cpu().numpy()
            ret, ret_dict=pnp_pycolmap(_xy, _coords, image_h, image_w, camera_type, camera_params, ransac_thresh)
            if ret_dict['success']:
                _coords=_coords[ret_dict['inliers']]
                _xy=_xy[ret_dict['inliers']]
                _coords_list.append(_coords)
                _xy_list.append(_xy)
    if len(_xy_list)>0:
        _xy=np.concatenate(_xy_list)
        _coords=np.concatenate(_coords_list)
        ret, ret_dict=pnp_pycolmap(_xy, _coords, image_h, image_w, camera_type, camera_params, ransac_thresh)
    return ret, ret_dict


def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    data.update({'R_errs_abs': [], 't_errs_abs': [], 'inliers': []})

    xy = data['xy_sample']
    
    camera_params = data['camera_params'].cpu().numpy()
    camera_type=data['camera_type']
    scale_factor=data['scale_factor'].cpu().numpy()
    pose = data['gt_pose'].cpu().numpy()
    
    coords=data['coords']
    N, C, H, W=data['image'].shape
    image_h=data['image_h']
    image_w=data['image_w']
    scores=data['scores']
    for bs in range(N):
        if config['trainer']['union_coords']: # Union all the coordiantes together and compute the pose once
            score=scores[bs, :]
            prediction=score>config['trainer']['score_thresh']
            _coords=coords[bs].cpu().numpy()
            _xy=xy[bs].cpu().numpy()
            _mask=prediction.cpu().numpy()
            _coords=_coords[_mask]
            _xy=_xy[_mask]/scale_factor[bs]
            ret, ret_dict=pnp_pycolmap(_xy, _coords, image_h[bs], image_w[bs], camera_type[bs], camera_params[bs], config['trainer']['ransac_thresh'])
        else: # Compute pose in each voxel, and merge the inlier points to compute the final pose
            ret, ret_dict = compute_pose(data['coords_list'], data['scores_list'], data['xy_list'], scale_factor[bs], config['trainer']['score_thresh'], bs, image_h[bs], image_w[bs], camera_type[bs], camera_params[bs], config['trainer']['ransac_thresh'])#pnp_pycolmap(_xy, _coords, image_h[bs], image_w[bs], camera_type[bs], camera_params[bs], config['trainer']['ransac_thresh'])

        if ret is None:
            data['R_errs_abs'].append(np.inf)
            data['t_errs_abs'].append(np.inf)
            data['inliers'].append(0)
            data['qvec']=np.zeros(4)
            data['tvec']=np.zeros(3)
        else:
            R=ret[:3,:3]
            t=ret[:3,3]
            t_err, R_err, t_err_abs, R_err_abs = relative_pose_error(pose[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs_abs'].append(R_err_abs)
            data['t_errs_abs'].append(t_err_abs)
            data['inliers'].append(sum(ret_dict['inliers']))
            data['qvec']=ret_dict['qvec']
            data['tvec']=ret_dict['tvec']


def aggregate_metrics(metrics):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')
    
    acc1=0
    acc2=0
    acc3=0
    for i, (r_err, t_err) in enumerate(zip(metrics['R_errs_abs'], metrics['t_errs_abs'])):
        if r_err<1 and t_err<0.1:
            acc1+=1
        if r_err<2 and t_err<0.25:
            acc2+=1
        if r_err<5 and t_err<1:
            acc3+=1
    acc1=acc1/len(metrics['R_errs_abs'])
    acc2=acc2/len(metrics['R_errs_abs'])
    acc3=acc3/len(metrics['R_errs_abs'])
   
    ret={
        't_median': np.median(metrics['t_errs_abs']),
        'r_median': np.median(metrics['R_errs_abs']),
        'acc1': acc1,
        'acc2': acc2,
        'acc3': acc3
    }
   
    return ret

def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [0.0,                 0.0,                 0.0, 1.0]])

def convert_pose(qvec, tvec):
    Tcw=quaternion_matrix(qvec)
    Tcw[:3,3]=tvec
    R=Tcw[:3,:3]
    T=tvec
    R_inv=R.T
    tw_inv = -R_inv.dot(T)
    Tcw_inv=np.eye(4)
    Tcw_inv[:3,:3]=R_inv
    Tcw_inv[:3,3]=tw_inv
    return Tcw_inv

def pnp_pycolmap(all_mkpq, all_mkp3d, height, width, camera_type, camera_params, ransac_thresh=48):
    cfg = {
            'model': camera_type,
            'width': width,
            'height': height,
            'params': list(camera_params)
        }

    ret = pycolmap.absolute_pose_estimation(all_mkpq, all_mkp3d, cfg, ransac_thresh)
    
    if ret['success']:
        ret['cfg'] = cfg
        qvec=ret['qvec']
        tvec=ret['tvec']
    else:
        return None, 0
    
    return convert_pose(qvec, tvec), ret