import imp
from multiprocessing.spawn import import_main_path
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['loss']  # config under the global namespace
 
    def compute_coordinate_loss(self, coords, coords_gt, mask): 
        mask=mask.reshape(-1)
        
        loss=torch.norm(coords-coords_gt, dim=2).reshape(-1)/self.config['scale']

        loss=loss[mask]
       
        loss_l1 = loss[loss <= 25]
        loss_sqrt = loss[loss > 25]
        loss_sqrt = torch.sqrt(25 * loss_sqrt)
        
        robust_loss = (loss_l1.sum() + loss_sqrt.sum()) / (float(loss.size(0))+1e-9)
        
        return robust_loss
    
    def binary_cls_loss(self, scores, mask, H, W):
        loss_func=torch.nn.BCEWithLogitsLoss()
        loss=loss_func(scores, mask.float())
        return loss

    def sparsity_loss(self, masks):
        loss=0
        for mask in masks:
            loss+=mask.abs().mean()

        loss/=len(masks)
        return loss
    
    def forward(self, data):
        
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        
        loss_scalars = {}
        N, _, H, W=data['image'].shape
        
        loss_c = self.compute_coordinate_loss(data['coords'], data['coords_gt_sample'], data['mask_sample'])
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
        
        loss_ce=self.binary_cls_loss(data['scores_logit'], data['mask_sample'], H, W)
        loss_scalars.update({"loss_ce": loss_ce.clone().detach().cpu()})
        
        loss_sparsity=self.sparsity_loss(data['embed_masks']) if 'embed_masks' in data else torch.tensor(0., device=loss_c.device)
        loss_scalars.update({"loss_sparsity": loss_sparsity.clone().detach().cpu() if 'embed_masks' in data else torch.tensor(0.)})
       
        if 'masks' in data:
            mask_num=0
            for mask in data['masks']:
                mask_num+=mask.sum()
            mask_num=mask_num/(len(data['masks'])*data['masks'][0].shape[0])
            loss_scalars.update({"mask_num": mask_num.clone().detach().cpu()})

        loss=loss_c+loss_ce*self.config['ce_scale']
        if self.config['sparsity_loss']:
            loss=loss+loss_sparsity

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})

