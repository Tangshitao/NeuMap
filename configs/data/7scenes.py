from configs.data.base import cfg

TEST_BASE_PATH = "data/kapture"
TRAIN_BASE_PATH = "data/kapture"

cfg.DATASET.TRAINVAL_DATA_SOURCE = "kapture"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/train_list/train_7scenes_v3.txt"
cfg.DATASET.TRAIN_SUBDIR='mapping'

cfg.DATASET.TEST_DATA_SOURCE = "kapture"
cfg.DATASET.VAL_DATA_ROOT= cfg.DATASET.TEST_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.VAL_LIST_PATH = f"{TEST_BASE_PATH}/train_list/query_heads_v3.txt"
cfg.DATASET.VAL_SUBDIR='query'

cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/train_list/query_heads_v3.txt"   
cfg.DATASET.TEST_SUBDIR='query'