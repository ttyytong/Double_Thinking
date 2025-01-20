#! /bin/bash
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 指定使用的GPU卡号
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置每个worker使用的GPU数量
NUM_GPUS_PER_WORKER=4

# NUM_GPUS_PER_WORKER=8
MP_SIZE=4

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="/hdd0/tyt/checkpoints/merged_model_490"
VERSION="base"
MODEL_ARGS="--from_pretrained /hdd0/tyt/checkpoints/checkpoints-4_8_0/merged_model_490 \
    --max_length 1288 \
    --lora_rank 10 \
    --use_lora \
    --local_tokenizer /hdd1/huggingface/models/lmsys/vicuna-7b-v1.5 \
    --version $VERSION"
# Tips: If training models of resolution 244, you can set --max_length smaller 


OPTIONS_SAT="SAT_HOME=/ssd0/tyt/CogVLM/checkpoints"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

train_data="/hdd0/tyt/datasets/construction/test_split/train"
test_data="/hdd0/tyt/datasets/construction/test_split/test"

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 0 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --test-data ${test_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 200 \
       --eval-interval 200 \
       --save "/hdd0/tyt/checkpoints" \
       --strict-eval \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config /ssd0/tyt/CogVLM/finetune_demo/test_config_bf16.json \
       --skip-init \
       --seed 2023
"

              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} /ssd0/tyt/CogVLM/finetune_demo/evaluate_cogvlm_demo.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x