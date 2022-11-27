import torch
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-m', '--model_path', type=str, help='model path')
parser.add_argument(
    '-s', '--scene', type=str, help='model path')
args=parser.parse_args()

path=args.model_path

state_dict=torch.load(args.model_path, map_location='cpu')['state_dict']

region_list=[]
with open('data/kapture/train_list/train_cambridge.txt') as f:
    for line in f.readlines():
        region_list.append(line.strip('\n'))
region_list=sorted(region_list)
token_num=0
for k, v in state_dict.items():
    if 'hard' in k:
        for i in range(v.shape[0]):
            if args.scene in region_list[i]:
                token_num+=v[i].sum()
print('Data size: {} MB'.format(token_num*256*4/1000000, token_num))
