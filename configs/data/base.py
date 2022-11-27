"""
The data config will be the last one merged into the main config.
Setups in data configs will override all existed setups!
"""

from yacs.config import CfgNode as CN
_CN = CN()
_CN.DATASET = CN()
_CN.TRAINER = CN()

# training data config
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
# validation set config
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None

# testing data config
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None



cfg = _CN
