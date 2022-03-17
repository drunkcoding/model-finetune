#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```

#SBATCH --mail-user=leyang.xue@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --partition=small
#SBATCH --cpus-per-task=10

# ====================
# Options for sbatch
# ====================
# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Megabytes of RAM required. Check `cluster-status` for node configurations
# #SBATCH --mem=32000

# Number of CPUs to use. Check `cluster-status` for node configurations
# #SBATCH --cpus-per-task=8

# Maximum time for the job to run, format: days-hours:minutes:seconds
# #SBATCH --time=1-00:00:00

# #SBATCH --gres=gpu:2
# =====================
# Logging information
# =====================
# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================
echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
# source ~/.bashrc
module load python/anaconda3
module load gcc
source activate torch

# Make script bail out after first error
set -e

task_name="squad_v2_moq"
model_name="bert-large-uncased"
base_dir=${HOME}/HuggingFace
batch_size=16
learning_rate=3e-5

mkdir -p ./outputs/${model_name}/${task_name}/
mkdir -p ./log/${model_name}/${task_name}/

deepspeed bert_train/run_qa.py \
    --deepspeed deepspeed_cfg_auto_stage0.json \
    --model_name_or_path ${base_dir}/${model_name} \
    --dataset_name squad_v2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --do_train \
    --version_2_with_negative \
    --per_device_train_batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --num_train_epochs 2 \
    --save_strategy epoch \
    --gradient_accumulation_steps 1 \
    --warmup_steps 100 \
    --weight_decay 0.0 \
    --overwrite_output_dir \
    --output_dir ./outputs/${model_name}/${task_name}/ \
    &> ./log/${model_name}/${task_name}/squad_bsz${batch_size}_lr${learning_rate}.log

# deepspeed bert_train/run_qa.py \
#     --deepspeed deepspeed_cfg_auto_stage0.json \
#     --model_name_or_path ${base_dir}/${model_name} \
#     --dataset_name squad_v2 \
#     --max_seq_length 384 \
#     --doc_stride 128 \
#     --do_train \
#     --version_2_with_negative \
#     --per_device_train_batch_size ${batch_size} \
#     --learning_rate ${learning_rate} \
#     --num_train_epochs 10 \
#     --save_strategy epoch \
#     --gradient_accumulation_steps 1 \
#     --warmup_steps 100 \
#     --weight_decay 0.0 \
#     --overwrite_output_dir \
#     --output_dir ./outputs/${model_name}/${task_name}/ \
#     &> ./log/${model_name}/${task_name}/squad_bsz${batch_size}_lr${learning_rate}.log

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

