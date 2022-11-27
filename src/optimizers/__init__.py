import torch
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR, StepLR


def build_optimizer(model, config):
    name = config['trainer']['optimizer']
    lr = config['trainer']['true_lr']
    backbone_lr=config['trainer']['backbone_lr']

    if name == "adamw":
        embeds_params=[]
        others=[]
        backbones=[]
        for k, v in model.named_parameters():
            if 'embeds' in k: # code parameters
                embeds_params.append(v) 
            elif 'backbone' in k: # backbone parameters
                backbones.append(v)
            else: # other parameters
                others.append(v)

        params=[{
            'params': embeds_params,
            'lr': lr*config['trainer']['code_lr_scale'],
            'weight_decay':config['trainer']['adamw_decay']
        }] # Code parameters must be optimized
        if not config['trainer']['code_finetune']: # Only code parameters are optimized if doing code finetune
            params.append({
                        'params': others,
                        'lr': lr,
                        'weight_decay':config['trainer']['adamw_decay']
                    }
                ) # transformer parameter must be optimized if not doing code finetune
            if not config['model']['backbone_freeze']: # freeze the backbone if pretrained backbone weights are available
                params.append({
                        'params': backbones,
                        'lr': backbone_lr,
                        'weight_decay':config['trainer']['adamw_decay']
                    }
                )
        return torch.optim.AdamW(params)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config['trainer']['scheduler_interval']}
    name = config['trainer']['scheduler']

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config['trainer']['mslr_milestones'], gamma=config['trainer']['mslr_gamma'])})
    else:
        raise NotImplementedError()

    return scheduler
