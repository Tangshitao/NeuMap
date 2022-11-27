import torch
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-m', '--model_path', type=str, help='model path')
args=parser.parse_args()

path=args.model_path

state_dict=torch.load(path)['state_dict']
weights=[]
params=0
for i in range(6):
    embeds=state_dict['model.embeds.{}.weight'.format(i)]
    if 'model.embed_mask_hard{}'.format(i) in state_dict:
        mask=state_dict['model.embed_mask_hard{}'.format(i)]
        b, n=mask.shape
        embeds=embeds.reshape(b, n, -1)
        embeds=embeds[mask]
    weights.append(embeds)
    params+=embeds.shape[0]*embeds.shape[1]
print('Data size: {} MB'.format(params*32/8/1000000))