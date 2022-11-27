
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import os
import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.NeuMap import NeuMap
from src.losses.loss import Loss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_NeuMap(pl.LightningModule):
    def __init__(self, config, backbone_pretrained_ckpt=None, neumap_pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        

        self.config = lower_config(config)  # full config
        self.save_hyperparameters(self.config)

        self.profiler = profiler or PassThroughProfiler()
        self.dump_dir = dump_dir
        # Matcher: LoFTR

        self.model = NeuMap(config=self.config)
      
        self.loss = Loss(self.config)

        if backbone_pretrained_ckpt is not None: # load backbone weights if there exists
            if os.path.exists(backbone_pretrained_ckpt):
                self.load_backbone_weight(backbone_pretrained_ckpt)
            else:
                raise Exception("backbone weights not exists: {}".format(backbone_pretrained_ckpt))
        
        if neumap_pretrained_ckpt is not None: # load neumap weights, used in code prune and code finetune
            if os.path.exists(neumap_pretrained_ckpt):
                self.load_neumap_weights(neumap_pretrained_ckpt)
            else:
                raise Exception("neumap weights not exists: {}".format(neumap_pretrained_ckpt))
        
        if self.config['trainer']['prune_finetune']: # register a hard mask if code pruning
            self.model.register_mask()
        # Testing
        
        # Pretrained weights
    def load_backbone_weight(self, backbone_pretrained_ckpt):

        state_dict = torch.load(backbone_pretrained_ckpt, map_location='cpu')['state_dict']
        pretrained_dict=self.model.state_dict()
        model_dict={}
        for k,v in state_dict.items():
            if k in pretrained_dict and v.shape==pretrained_dict[k].shape:
                model_dict[k]=v
                print('hit', k)
            elif k[:8]=='backbone':
                print('miss', k)
        self.model.load_state_dict(model_dict, strict=False)
        logger.info(f"Load \'{backbone_pretrained_ckpt}\' as pretrained checkpoint")
    
    def load_neumap_weights(self, neumap_pretrained_ckpt):
        state_dict=torch.load(neumap_pretrained_ckpt, map_location='cpu')['state_dict']
        def check_key(key):
            if self.config['trainer']['code_finetune'] and ('embed' in key or '.max' in key or '.min' in key or '.mean' in key or '.std' in key):
                return True
            else:
                return False
        state_dict={k: v for k, v in state_dict.items() if not check_key(k)}
        self.load_state_dict(state_dict, strict=False)
        
    def configure_optimizers(self):
        optimizer1 = build_optimizer(self, self.config)
        scheduler1 = build_scheduler(self.config, optimizer1)

        return [optimizer1], [scheduler1]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
       
        warmup_step = self.config['trainer']['warmup_step']
        if self.trainer.global_step < warmup_step:
            if self.config['trainer']['warmup_type'] == 'linear':
                base_lr = self.config['trainer']['warmup_ratio'] * self.config['trainer']['true_lr']
                lr = base_lr + \
                    (self.trainer.global_step / self.config['trainer']['warmup_step']) * \
                    abs(self.config['trainer']['true_lr'] - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            else:
                raise ValueError('Unknown lr warm-up strategy: {}'.format(self.config['trainer']['warmup_type']))
        
        # update params
        optimizer.step(closure=optimizer_closure)
        
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):    
        with self.profiler.profile("NeuMap"):
            self.model(batch)
    
        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair
            
            image_name = list(zip(*batch['image_name']))
            bs = batch['image'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(image_name[b]) for b in range(bs)],
                'inliers': batch['inliers'],
                'R_errs_abs': batch['R_errs_abs'],
                't_errs_abs': batch['t_errs_abs'],
                'qvec': batch['qvec'],
                'tvec': batch['tvec'],
                'name': batch['image_name']}
            ret_dict = {'metrics': metrics}
        return ret_dict
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.log(f'train/{k}', v, self.global_step)
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.log(
                'train/avg_loss_on_epoch', avg_loss)
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        ret_dict = self._compute_metrics(batch)
        
        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
        }
        
    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics(metrics)
        
            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.log(f'val_{valset_idx}/avg_{k}', mean_v)

                for k, v in val_metrics_4tb.items():
                    self.log(f"metrics_{valset_idx}/{k}", v)
            plt.close('all')

        self.log(f'r_median', torch.tensor(val_metrics_4tb['r_median']))
        self.log(f't_median', torch.tensor(val_metrics_4tb['t_median']))
        self.log(f'acc1', torch.tensor(val_metrics_4tb['acc1']))
        self.log(f'acc2', torch.tensor(val_metrics_4tb['acc2']))
        self.log(f'acc3', torch.tensor(val_metrics_4tb['acc3']))
     
    def test_step(self, batch, batch_idx): 
        with self.profiler.profile("LoFTR"):
            self.model(batch)


        with self.profiler.profile("Compute losses"):
            self.loss(batch)
        
        ret_dict = self._compute_metrics(batch)
        
        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                image_name = list(zip(*batch['image_name']))
                bs = batch['image'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    item['image_name'] = image_name[b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps
        
        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy

        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
           
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'NSM_pred_eval', dumps)
            names_set=set()
            
            with open(os.path.join(self.dump_dir, 'pose.txt'), 'w') as f:
                for i, name in enumerate(metrics['name']):
                    if name in names_set:
                        continue
                    names_set.add(name)
                    qvec=metrics['qvec'][i*4:i*4+4]
                    tvec=metrics['tvec'][i*3:i*3+3]
                    qvec = ' '.join(map(str, qvec))
                    tvec = ' '.join(map(str, tvec))
                    f.write(f'{name} {qvec} {tvec}\n')
                   