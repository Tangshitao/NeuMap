#!/bin/bash -l
# a indoor_ds model with the pos_enc impl bug fixed.

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/7scenes.py"
main_cfg_path="configs/neumap/7scenes_stage2.py"
ckpt_path="model_release/7scenes.ckpt"
dump_dir="dump/7scenes"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=1
torch_num_workers=0
batch_size=1  # per gpu

python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark \
    --ckpt_path $ckpt_path
    