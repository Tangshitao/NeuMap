from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  LoFTR Pipeline  ↓  ##############
_CN.MODEL = CN()
_CN.MODEL.RGB=False #Use RGB image or Grey image
_CN.MODEL.BACKBONE_FREEZE=True #Freeze the backbone weight, used in code finetune or pretrain backbone weight

# 1. LoFTR-backbone (local feature CNN) config
_CN.MODEL.RESNETFPN = CN()
_CN.MODEL.RESNETFPN.INITIAL_DIM = 128 #Dimensions after the first convolutional layer 
_CN.MODEL.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3
_CN.MODEL.RESNETFPN.STRIDES = [1, 2, 2]
_CN.MODEL.NHEAD = 4 #Transformer heads
_CN.MODEL.FFDIM_FACTOR = 4 #MLP middle dimensions
_CN.MODEL.TRANS_BLOCK_NUM=1 #Number of transformer blocks
_CN.MODEL.CODE_NUM = 100 #Code number per transformer block
_CN.MODEL.D_MODEL = 100 #
_CN.MODEL.N_SAMPLE_POINTS = 5000 #Sample N points for each image in the training stage
_CN.MODEL.N_SAMPLE_IN_VOXEL_POINTS = 4000 #Sample M in-voxels points for each image

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['scannet', '7scenes', 'aachen', 'NL', 'cambridge']
_CN.DATASET.TRAIN_DATA_ROOT = None # Root folder
_CN.DATASET.TRAIN_LIST_PATH = None # Training reegion list path
_CN.DATASET.TRAIN_SUBDIR='mapping' # 'mapping'
_CN.DATASET.VAL_DATA_ROOT = None # 'validation', 'testing'
_CN.DATASET.VAL_LIST_PATH = None   # Validation reegion list path
_CN.DATASET.VAL_SUBDIR='mapping' # 'validation', 'testing'
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None
_CN.DATASET.RESOLUTION=640
_CN.DATASET.TEST_SUBDIR='query'
_CN.DATASET.MAX_N_POINTS=20000 # Sample maximum N points in dataloader
_CN.DATASET.RANDOM_CROP=True 
_CN.DATASET.ASPECT_RATIO=None 
# 2. dataset config



##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adamw]
_CN.TRAINER.TRUE_LR = None  # Initial learning rate
_CN.TRAINER.ADAMW_DECAY = 0.
_CN.TRAINER.CODE_FINETUNE=False # Reinitalize and finetune code
# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 2000
_CN.TRAINER.PRUNE_FINETUNE=False #Code prune
_CN.TRAINER.PRUNE_THRESH=0.1 #Code pruning thresh

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5


# geometric metrics and pose solver
_CN.TRAINER.SCORE_THRESH=0.5 #Keep predictions whose 
_CN.TRAINER.RANSAC_THRESH=48 #RANSAC reporjection error threshold

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 256 # Sample n images for each voxel


# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5
_CN.TRAINER.LRSTEPS=700
_CN.TRAINER.BACKBONE_LR=2e-3 # Backbone initial learning rate
_CN.TRAINER.CODE_LR_SCALE=0.05 # CODE_LR=TRUE_LR*CODE_LR_SCALE
_CN.TRAINER.UNION_COORDS=False
# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66

_CN.LOSS = CN()
_CN.LOSS.COORD_LOSS_THRESH=25 #L2 loss if loss<COORD_LOSS_THRESH, else square root loss
_CN.LOSS.SCALE=1.0 #
_CN.LOSS.SPARSITY_LOSS=False #Sparsity loss
_CN.LOSS.CE_SCALE=1.0 #BCE loss scale
def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
