task_name=$1
model_name=$2
base_dir=$4
batch_size=16
learning_rate=$3

LOG_DIR="./log/${model_name}/${task_name}/"
OUT_DIR="./outputs/${model_name}/${task_name}/"

export TORCH_EXTENSIONS_DIR=/jmain02/home/J2AD002/jxm12/lxx22-jxm12/model-finetune

mkdir -p ${OUT_DIR}
mkdir -p ${LOG_DIR}

# ~/.conda/envs/torch/bin/python huggingface-vit-finetune/run_image_classification.py \
deepspeed huggingface-vit-finetune/run_image_classification.py \
    --output_dir ${OUT_DIR} \
    --deepspeed deepspeed_cfg_auto_stage0_vit.json \
    --model_name_or_path ${base_dir}/${model_name} \
    --dataset_name glue \
    --do_train \
    --fp16 \
    --per_device_train_batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --num_train_epochs 10 \
    --save_strategy steps \
    --logging_strategy steps \
    --logging_steps 500 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 100 \
    --weight_decay 0.0  \
    --overwrite_output_dir \
    &> ${LOG_DIR}/vit_bsz${batch_size}_lr${learning_rate}.log