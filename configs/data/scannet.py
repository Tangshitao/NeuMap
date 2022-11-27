from configs.data.base import cfg

TEST_BASE_PATH = "data/scannet"
TRAIN_BASE_PATH = "data/scannet"

cfg.DATASET.TRAINVAL_DATA_SOURCE = "scannet"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/train_list/train_scannet_80.txt"

cfg.DATASET.TEST_DATA_SOURCE = "scannet"
cfg.DATASET.VAL_DATA_ROOT= cfg.DATASET.TEST_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.VAL_LIST_PATH = f"{TEST_BASE_PATH}/train_list/query_scannet_1.txt"

cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/train_list/query_scannet_1.txt"   

