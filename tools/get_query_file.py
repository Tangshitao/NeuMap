import os
import numpy as np
import argparse


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-root', '--root', type=str, help='data root')
    parser.add_argument(
        '-retrieval_path', '--retrieval_path', type=str, help='retrival path')
    parser.add_argument(
        '-train_list_path', '--train_list_path', type=str, help='train list path')
    parser.add_argument(
        '-o', '--output', type=str, help='output path')
   
    return parser.parse_args()

args=parse_args()
retrieval_path=args.retrieval_path
data_dir=args.root


image_idx_map={}

with open(args.train_list_path) as ff:
    for line1 in ff.readlines():
        path=os.path.join(args.root, line1.strip('\n'))
        info=np.load(path, allow_pickle=True).item()
        image_names=info['image_names']
        for name in image_names:
            if name not in image_idx_map:
                image_idx_map[name]=[]
            image_idx_map[name].append(line1.strip('\n'))
            voxel_id=line1.strip('\n')

query_voxel_map={}
with open(args.retrieval_path) as f:
    for i, line in enumerate(f.readlines()):
       
        query_fn, scene_fn=line.strip('\n').split(' ')
        query_fn=query_fn[6:]
        if query_fn not in query_voxel_map:
            query_voxel_map[query_fn]=set()
        if scene_fn in image_idx_map:
            query_voxel_map[query_fn]= query_voxel_map[query_fn].union(image_idx_map[scene_fn]) 

tot=0
for k, v in query_voxel_map.items():
    print(k, len(v))
    tot+=len(v)

voxel_id=os.path.dirname(voxel_id)
np.save(os.path.join(args.root, voxel_id, args.output+'.npy'), query_voxel_map)
with open(os.path.join(args.root, 'train_list', args.output+'.txt'), 'w') as f:
    f.write(os.path.join(voxel_id, args.output+'.npy'))

