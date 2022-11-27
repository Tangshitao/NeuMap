from configs.data.base import cfg

TEST_BASE_PATH = "data/kapture"
TRAIN_BASE_PATH = "data/kapture"

cfg.DATASET.TRAINVAL_DATA_SOURCE = "kapture"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/train_list/train_regular_3_Store_B1.txt"
cfg.DATASET.TRAIN_INTRINSIC_PATH = f"{TRAIN_BASE_PATH}/intrinsics.npz"
cfg.DATASET.TRAIN_SUBDIR='mapping'

cfg.DATASET.TEST_DATA_SOURCE = "kapture"
cfg.DATASET.VAL_DATA_ROOT= cfg.DATASET.TEST_DATA_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.VAL_LIST_PATH = f"{TEST_BASE_PATH}/train_list/val_regular_3_Store_B1_sampled.txt"
cfg.DATASET.VAL_SUBDIR='validation'

cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/train_list/val_regular_3_Store_B1.txt"   
cfg.DATASET.TEST_SUBDIR='validation'

