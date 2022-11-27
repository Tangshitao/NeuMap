import os
import copy
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

from .backbone import build_backbone
from .backbone.resnet_fpn import BasicBlock
from .transformer_module.vit import Block


class NeuMap(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        self.n_points= config['model']['n_sample_points'] # sample N points to compute loss
        self.n_in_voxel_points= config['model']['n_sample_in_voxel_points'] # sample M points to compute coordinate loss
        self.backbone_freeze=config['model']['backbone_freeze'] # freeze the backbone
        self.code_finetune=config['trainer']['code_finetune'] # fintune code, make sure to free other parameters

        self.prune_finetune=config['trainer']['prune_finetune'] # prune codes, and finetune the reset
        self.prune_thresh=config['trainer']['prune_thresh'] # prune codes larger than threshold

        # Modules
        self.load_scene_info(config['dataset']) # load scene list

        self.backbone=build_backbone(config['model']) # set up backbone
        self.d_model=self.backbone.dim 

        n_blocks=config['model']['trans_block_num']
        
        self.transformer=nn.ModuleList(
            [Block(self.d_model, config['model']['nhead'], \
            self.d_model*config['model']['ffdim_factor']) for i in range(n_blocks)]
        ) # set up transformer block

        self.res1=BasicBlock(self.d_model, self.d_model) #Addtional backbone block 1, 
        self.res2=BasicBlock(self.d_model, self.d_model) #Addtional backbone block 2, this 2 blocks may be useless, but it's legacy, feel free to remove it
        self.final_conv=nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0, bias=True) # The last backbone conv layer
        
        self.code_num=config['model']['code_num']
        embeddings=nn.Embedding(len(self.name_to_idx), self.code_num*self.d_model) # set up codes
        self.embeds = nn.ModuleList([copy.deepcopy(embeddings) for _ in range(n_blocks)]) # copy codes for each scene
        for i in range(n_blocks):
            self.register_parameter('embed_mask{}'.format(i), nn.Parameter(torch.ones(len(self.name_to_idx), self.code_num))) # sparsity weight

        self.out_conv=nn.Linear(self.backbone.dim, 4) # output layer, (x, y, z, c)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def register_mask(self): # register a hard weight for pruning finetune
        for i in range(self.config['model']['trans_block_num']):
            embed_mask=getattr(self, 'embed_mask{}'.format(i)).abs()
            max_idx=embed_mask.argmax(dim=1) # At least 1 points are kepts
            embed_mask=embed_mask>self.prune_thresh
            idx=torch.arange(0, embed_mask.shape[0])
            embed_mask[idx, max_idx]=True
            self.register_buffer('embed_mask_hard{}'.format(i), embed_mask)

    def load_scene_info(self, config):
        self.name_to_idx={}
        name_set=set()
        
        with open(config['train_list_path']) as f:
            for idx, line in enumerate(f.readlines()):
                scene=line.strip("\n")
                name_set.add(scene)

        idx=0
        means=[]
        mins=[]
        maxs=[]
        stds=[]

        for name in sorted(list(name_set)):
            self.name_to_idx[name]=idx
            points_path=os.path.join(config['train_data_root'], name)
            points_info=np.load(points_path, allow_pickle=True).item()
           
            means.append(points_info['xyz_median'][:3])
            stds.append(points_info['xyz_std'][:3])
            mins.append(points_info['xyz_min'][:3])
            maxs.append(points_info['xyz_max'][:3])
            idx+=1
        
        self.register_buffer('mean', torch.tensor(np.stack(means, axis=0)))
        self.register_buffer('min', torch.tensor(np.stack(mins, axis=0)))
        self.register_buffer('max', torch.tensor(np.stack(maxs, axis=0)))
        self.register_buffer('std', torch.tensor(np.stack(stds, axis=0)))

    def sample_in_voxel_points(self, xy, mask, pts3D, H, W): 
        N=xy.shape[0]
        new_xy=torch.zeros(N, self.n_points, 2, device=xy.device)
        new_mask=torch.zeros(N, self.n_points, device=xy.device).bool()
        new_pts3D=torch.zeros(N, self.n_points, 3, device=xy.device)
       
        xy_int=torch.round(xy).long()
        
        xy_int[:, :, 0]=torch.clamp(xy_int[:, :, 0], min=0, max=W-1)
        xy_int[:, :, 1]=torch.clamp(xy_int[:, :, 1], min=0, max=H-1)

        for n in range(xy.shape[0]):
            mark=torch.zeros(H, W, device=xy.device)
            xy_iv=xy[n, mask[n]]
            xy_iv_int=xy_int[n, mask[n]]
            pts_iv=pts3D[n, mask[n]]
            
            for i in range(-5, 5): # Those points are not sampled as out of voxel points
                for j in range(-5, 5):
                    x=torch.clamp(xy_iv_int[:, 0]+i, min=0, max=W-1)
                    y=torch.clamp(xy_iv_int[:, 1]+j, min=0, max=H-1)
                    mark[y,x]=1
            or_mask=(1-mark)[xy_int[n, :,1], xy_int[n, :, 0]].bool() # masks of out-of-voxel points
            ov_idx=list(torch.where(or_mask)[0].cpu().numpy()) # index of out-of-voxel points
            random.shuffle(ov_idx)

            iv_idx=list(range(xy_iv_int.shape[0])) # index of in-voxel points
            random.shuffle(iv_idx)
            xy_iv=xy_iv[iv_idx] # pixel coordinates of in-voxel points
            pts_iv=pts_iv[iv_idx] # 3D coordinates of in-voxel points

            n_in_voxel_points=min(self.n_in_voxel_points, xy_iv.shape[0]) # sample N in-voxel points
            n_out_voxel_points=min(self.n_points-n_in_voxel_points, len(ov_idx)) # sample M out-voxel points
            ov_idx=ov_idx[:n_out_voxel_points] # index of N out-voxel points
            new_xy[n,:n_in_voxel_points]=xy_iv[:n_in_voxel_points] # Top N points are in-voxel
            
            new_xy[n,n_in_voxel_points:n_in_voxel_points+n_out_voxel_points]=xy[n, ov_idx]
            new_pts3D[n, :n_in_voxel_points]=pts_iv[:n_in_voxel_points]
            new_mask[n, :n_in_voxel_points]=True
        
        return new_xy, new_mask, new_pts3D
         
    def get_codes(self, voxel_ids, device):
        voxel_codes=[]
        voxel_code_soft_masks=[] # sparsity score 
        voxel_code_hard_masks=[] # sparsity mask
        codes=self.embeds
        for code_layer_i, code in enumerate(codes):
            code_one_layer=[]
            mask_one_layer=[]
            for i, v_id in enumerate(voxel_ids):
                
                idx=self.name_to_idx[v_id] # code index
                idx_gpu=torch.tensor(idx, dtype=torch.long, device=device)	
                code_soft_mask=getattr(self, 'embed_mask{}'.format(code_layer_i))[idx]

                code_single_voxel=code(idx_gpu).reshape(-1, self.d_model)*code_soft_mask[:,None]
              
               
                if self.prune_finetune: # get hard mask if doing pruning
                    code_hard_mask=getattr(self, 'embed_mask_hard{}'.format(code_layer_i))[idx] 
                else:
                    code_hard_mask=torch.ones(self.code_num, device=code_single_voxel.device).bool()

                code_one_layer.append(code_single_voxel)
                mask_one_layer.append(code_hard_mask)
                voxel_code_soft_masks.append(code_soft_mask)
                
            embeddings_one_layer=torch.stack(code_one_layer)
            masks_one_layer=torch.stack(mask_one_layer)
            voxel_code_hard_masks.append(masks_one_layer)
            voxel_codes.append(embeddings_one_layer)

        return voxel_codes, voxel_code_hard_masks, voxel_code_soft_masks
    
    def interporlate_feat(self, feat, keypoints, H, W):
        _, _, fH, fW=feat.shape
        kps=keypoints.clone()
        _H=H
        
        while _H!=fH:
            kps=(kps-0.5)/2
            _H=_H//2
        
        kps[:,:,0]=kps[:,:,0]/(fW-1)*2-1
        kps[:,:,1]=kps[:,:,1]/(fH-1)*2-1
        kps=kps[:, :, None, :]
       
        feat_sample=F.grid_sample(feat, kps, mode='bilinear', align_corners=True).squeeze(-1).permute(0, 2, 1)
    
        return feat_sample

    
    def get_coord_info(self, data):
        voxel_id=data['voxel_id']
        means=[]
        mins=[]
        maxs=[]
        stds=[]

        for i, s_id in enumerate(voxel_id):
            if s_id not in self.name_to_idx:
                idx=0
            else:
                if s_id not in self.name_to_idx:
                    idx=0
                else:
                    idx=self.name_to_idx[s_id]

            means.append(self.mean[idx])
            mins.append(self.min[idx])
            maxs.append(self.max[idx])
            stds.append(self.std[idx])
            
        data['mean']=torch.stack(means)
        data['min']=torch.stack(mins)
        data['max']=torch.stack(maxs)
        data['std']=torch.ones(len(voxel_id), 1, device=self.mean.device) # Useless, but we still keep it to make sure the weights are loaded successfully

       
    def train_forward(self, data):
        self.get_coord_info(data) # get mean, min, max coordinates in a voxel
        N, _, H, W=data['image'].shape
        
        codes, code_hard_masks, code_soft_masks=self.get_codes(data['voxel_id'], data['image'].device)
        data['code_hard_masks']=code_hard_masks
        data['code_soft_masks']=code_soft_masks

        if 'feat' not in data:
            
            if self.backbone_freeze or self.code_finetune: # if code finetune, freeze the backbone
                self.backbone.eval()
                with torch.no_grad():
                    feat=self.backbone(data['image'])
            else:
                feat=self.backbone(data['image'])
            
            if self.code_finetune:
                self.res1.eval()
                self.res2.eval()
            feat=self.res1(feat)
            feat=self.res2(feat)
            feat=self.final_conv(feat)
        else:
            feat=data['feat']
        
        mask=((data['gt_coords']>=data['min'][:,None,:])*(data['gt_coords']<=data['max'][:,None,:])).all(dim=-1)*data['mask']

        if self.training:
            xy, mask, coords_gt=self.sample_in_voxel_points(data['xy'], mask, data['gt_coords'], H, W)
        else:
            xy, mask, coords_gt=data['xy'], mask, data['gt_coords']

        feat=self.interporlate_feat(feat, xy, H, W) # features of key-pints
        
        for i, block in enumerate(self.transformer):
            feat=block(feat, codes[i], code_hard_masks[i]) 
        
        output=self.out_conv(feat)

        scores_logit=output[:,:,3]
        coords=output[:,:,:3]+data['mean'][:,None,:]

        data['coords']=coords # coordiante prediction
        data['scores_logit']=scores_logit # score prediction
        data['xy_sample']=xy # sampled key-points
        data['mask_sample']=mask # sampled mask
        data['coords_gt_sample']=coords_gt # sampled ground truth
       
    def forward(self, data):

        if self.training:
            self.train_forward(data)
        else:
            coords_list=[]
            scores_list=[]
            xy_sample_list=[]
            mask_list=[]
            coords_gt=[]
            with torch.no_grad(): # extract image features
                feat=self.backbone(data['image'])
                feat=self.res1(feat)
                feat=self.res2(feat)
                feat=self.final_conv(feat)
            for voxel_id in data['voxel_id']: # predict coordiantes voxel by voxel
                data_copy=copy.deepcopy(data)
                data_copy['voxel_id']=voxel_id
                data['feat']=feat
                self.train_forward(data_copy)
                coords_list.append(data_copy['coords'])
                scores_list.append(data_copy['scores_logit'])
                xy_sample_list.append(data_copy['xy_sample'])
                mask_list.append(data_copy['mask_sample'])
                coords_gt.append(data_copy['coords_gt_sample'])

            data['coords_list']=coords_list
            data['scores_list']=[torch.sigmoid(scores) for scores in scores_list]
            data['xy_list']=xy_sample_list
            data['coords']=torch.cat(coords_list, dim=1) # coordinate prediction
            data['scores_logit']=torch.cat(scores_list, dim=1) # score prediction
            data['xy_sample']=torch.cat(xy_sample_list, dim=1) # pixel coordinate 
            data['mask_sample']=torch.cat(mask_list, dim=1)
            data['coords_gt_sample']=torch.cat(coords_gt, dim=1)
   
        data['scores']=torch.sigmoid(data['scores_logit'])
