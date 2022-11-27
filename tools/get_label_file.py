import numpy as np
import os
import tqdm
import random
import argparse
from kapture.io.csv import kapture_from_dir
import kapture

def load_points(kdata, data_dir):
    points_list=[]
    names=[]
    tot_pts=0
    max_list=[]
    min_list=[]
    points_dir=os.path.join(data_dir, 'points')
    for _, sensor_id, filename in kapture.flatten(kdata.records_camera, is_sorted=True):
        print('read {}'.format(filename))
        
        pts_path=os.path.join(points_dir, filename+'.npy')
        
        if not os.path.exists(pts_path):
            print(pts_path)
            continue
        info=np.load(pts_path, allow_pickle=True).item()
        points=info['points']
        mask=info['mask']
        if mask.sum()==0:
            continue
        points=points[mask,:3]
        
        points_list.append(points[:,:3])
       
        tot_pts+=points.shape[0]
       
        names.append(filename)
        if points.shape[0]>0:
            _min=np.percentile(points, 1, axis=0)[:3]
            _max=np.percentile(points, 99, axis=0)[:3]
            max_list.append(_max)
            min_list.append(_min)
    
    return points_list, names, max_list, min_list


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-root', '--root', type=str, help='data root')
    parser.add_argument(
        '-i', '--input', type=str, help='input scene')
    parser.add_argument(
        '-s', '--size', type=int, help='voxel size')
    parser.add_argument(
        '-p', '--points_thresh', type=int, help='filter points', default=20)
    parser.add_argument(
        '-it', '--n_image_thresh', type=int, help='filter images', default=20)
    parser.add_argument(
        '-o', '--output', type=str, help='output name')
   
    return parser.parse_args()

args=parse_args()
path=os.path.join(args.root, args.input)
kdata=kapture_from_dir(path)
train_points_list, train_names, max_list, min_list=load_points(kdata, path) 
_max=np.stack(max_list)
_min=np.stack(min_list)
max_voxel=_max.max(axis=0)[:3]
min_voxel=_min.min(axis=0)[:3]

block_size=args.size
train_list_dict={}

points_thresh=args.points_thresh
for pidx, pts in enumerate(train_points_list):

    range_x_max=pts[:,0].max()
    range_x_min=pts[:,0].min()
    min_idx_x=int((range_x_min-min_voxel[0])//block_size)
    max_idx_x=int((range_x_max-min_voxel[0])//block_size)
    
    for idx_x in range(min_idx_x, max_idx_x+1):
        voxel_range_x=min_voxel[0]+idx_x*block_size
        mask_x=(pts[:,0]>=voxel_range_x)*(pts[:,0]<voxel_range_x+block_size)
        
        if mask_x.sum()<=points_thresh:
            continue
        pts_x=pts[mask_x]
        range_y_max=pts[:,1].max()
        range_y_min=pts[:,1].min()
        min_idx_y=int((range_y_min-min_voxel[1])//block_size)
        max_idx_y=int((range_y_max-min_voxel[1])//block_size)
        for idx_y in range(min_idx_y, max_idx_y+1):
            voxel_range_y=min_voxel[1]+idx_y*block_size
            mask_y=(pts_x[:,1]>=voxel_range_y)*(pts_x[:,1]<voxel_range_y+block_size)
            if mask_y.sum()<=points_thresh:
                continue
            pts_y=pts_x[mask_y]
            range_z_max=pts[:,2].max()
            range_z_min=pts[:,2].min()
            min_idx_z=int((range_z_min-min_voxel[2])//block_size)
            max_idx_z=int((range_z_max-min_voxel[2])//block_size)
            for idx_z in range(min_idx_z, max_idx_z+1):
                voxel_range_z=min_voxel[2]+idx_z*block_size
                mask_z=(pts_y[:,2]>=voxel_range_z)*(pts_y[:,2]<voxel_range_z+block_size)
                
                if mask_z.sum()>points_thresh:
                
                    if (idx_x, idx_y, idx_z) not in train_list_dict:
                        train_list_dict[(idx_x, idx_y, idx_z)]=[]
                        print(len(train_list_dict))
                    train_list_dict[(idx_x, idx_y, idx_z)].append((train_names[pidx], pts_y[mask_z]))

train_idx=0
train_fn_list=[]
n_image_thresh=args.n_image_thresh
middle_dir='train_list_{}'.format(block_size)
save_dir=os.path.join(os.path.dirname(path), middle_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for (idx_x, idx_y, idx_z), name_pts in train_list_dict.items():
    if len(name_pts)<n_image_thresh:
        continue
    name_list=[name for name, _ in name_pts]
    voxel_min=min_voxel+np.array((idx_x, idx_y, idx_z))*block_size
    voxel_max=voxel_min+block_size
    pts=np.concatenate([v[1] for v in name_pts])
    coord_mean=pts.mean(axis=0)
    coord_std=pts.std(axis=0)
    coord_median=np.median(pts, axis=0)
    info={
        'xyz_max': voxel_max,
        'xyz_min': voxel_min,
        'xyz_mean': coord_mean,
        'xyz_median': coord_median,
        'xyz_std': coord_std,
        'image_names': name_list
    }
    fn='train_all_{}_{}.npy'.format(block_size, train_idx)
    path=os.path.join(save_dir, fn)
    np.save(path, info)
    train_fn_list.append(fn)
    train_idx+=1

with open(os.path.join('data/kapture/train_list/{}.txt'.format(args.output)),'w') as f:
    for fn in train_fn_list:
        path=os.path.join(os.path.dirname(args.input), middle_dir, fn)
        f.write(path+'\n')
