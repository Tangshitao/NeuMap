import os
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader
)


from src.utils.dataloader import get_local_split
from src.datasets.scannet import ScanNetDataset
from src.datasets.kapture import KaptureDataset
from src.datasets.sampler import RandomConcatSampler
from kapture.io.csv import kapture_from_dir

class MultiSceneDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_subdir=config.DATASET.TRAIN_SUBDIR
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_subdir=config.DATASET.VAL_SUBDIR
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_subdir=config.DATASET.TEST_SUBDIR
        self.resolution=config.DATASET.RESOLUTION
        self.max_n_points=config.DATASET.MAX_N_POINTS
        self.random_crop=config.DATASET.RANDOM_CROP
        self.aspect_ratio=config.DATASET.ASPECT_RATIO
        self.rgb=config.MODEL.RGB

        # 2. dataset config
        # general options
       
        # 3.loader parameters
        self.batch_size=args.batch_size
        
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }

        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }
        
        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66
      

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_list_path,
                mode='train')
            
            self.val_dataset = self._setup_dataset(
                self.val_data_root,
                self.val_list_path,
                mode='val')
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_list_path,
                mode='test')
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_root,
                       scene_list_path,
                       mode='train'):
        """ Setup train / val / test set"""
        
        with open(scene_list_path, 'r') as f:
            train_list = [name.strip('\n') for name in f.readlines()]

        if mode == 'train':
            local_train_list = get_local_split(train_list, self.world_size, self.rank, self.seed)
        else:
            local_train_list = train_list
        logger.info(f'[rank {self.rank}]: {len(local_train_list)} scene(s) assigned.')
       
        dataset_builder = self._build_concat_dataset
        return dataset_builder(data_root, local_train_list, mode=mode)

    def _build_concat_dataset(
        self,
        data_root,
        train_list,
        mode
    ):
        
        datasets = []
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
    
        if data_source=='kapture' or data_source=='7scenes':
            kapture_datas={}
            sensor_datas={}
            input_path_datas={}
            train_list_kapture_map={}
            for train_path in train_list:
                scene=os.path.dirname(os.path.dirname(train_path))
                if scene not in kapture_datas:
                    if mode=='test':
                        input_path=os.path.join(data_root,scene, self.test_subdir)
                    elif mode=='train':
                        input_path=os.path.join(data_root,scene, self.train_subdir)
                    else:
                        input_path=os.path.join(data_root, scene, self.val_subdir)
                    kapture_data=kapture_from_dir(input_path)
                    sensor_dict={}
                    for timestep in kapture_data.records_camera:
                        _sensor_dict=kapture_data.records_camera[timestep]
                        for k, v in _sensor_dict.items():
                            sensor_dict[v]=(timestep, k)
                    kapture_datas[scene]=kapture_data
                    sensor_datas[scene]=sensor_dict
                    input_path_datas[scene]=input_path
                train_list_kapture_map[train_path]=(kapture_datas[scene], sensor_datas[scene], input_path_datas[scene])
       
        for train_path in tqdm(train_list,
                             desc=f'[rank:{self.rank}] loading {mode} datasets',
                             disable=int(self.rank) != 0):
            if data_source=='kapture':
                kapture_data, sensor_data, input_path=train_list_kapture_map[train_path]
                datasets.append(
                    KaptureDataset(
                            data_root, 
                            kapture_data=kapture_data, 
                            sensor_dict=sensor_data,
                            train_path=train_path, 
                            input_path=input_path,
                            mode=mode,
                            image_size=self.resolution,
                            max_n_points=self.max_n_points,
                            random_crop=self.random_crop,
                             rgb=self.rgb, 
                             aspect_ratio=self.aspect_ratio
                        )
                    )
            elif data_source=='7scenes':
                kapture_data, sensor_data, input_path=train_list_kapture_map[train_path]
                datasets.append(
                    SevenScenesDataset(
                            data_root, 
                            kapture_data=kapture_data, 
                            sensor_dict=sensor_data,
                            train_path=train_path, 
                            input_path=input_path,
                            mode=mode,
                            image_size=self.resolution,
                            max_n_points=self.max_n_points,
                            random_crop=self.random_crop,
                             rgb=self.rgb, 
                             aspect_ratio=self.aspect_ratio
                        )
                    )
            elif data_source=='scannet':
                datasets.append(
                    ScanNetDataset(
                        data_root,
                        train_path,
                        mode=mode,
                        image_size=self.resolution,
                        max_n_points=self.max_n_points,
                        rgb=self.rgb, 
                    )
                )
            else:
                raise NotImplementedError()
        return ConcatDataset(datasets)
    
    
    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        #assert self.data_sampler in ['scene_balance']
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        if self.data_sampler == 'scene_balance':
            sampler = RandomConcatSampler(self.train_dataset,
                                          self.n_samples_per_subset,
                                          self.seed, self.batch_size)
        else:
            sampler = RandomSampler(self.train_dataset)

        dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
        
        return dataloader
    
    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        if not isinstance(self.val_dataset, abc.Sequence):
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
            return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                sampler = DistributedSampler(dataset, shuffle=False)
                dataloaders.append(DataLoader(dataset, sampler=sampler, **self.val_loader_params))
            return dataloaders

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)

def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
