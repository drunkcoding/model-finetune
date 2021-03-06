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
#SBATCH --gres=gpu:8
#SBATCH --partition=big
#SBATCH --cpus-per-task=40

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
 
# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

# experiment_text_file=$1
# COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"

#source /etc/profile.d/modules.sh
#module load cuda

MODULE="google/t5-large-lm-adapt"
LR=6e-5
TASK="mnli"

echo "Job running ${MODULE}, ${LR}, ${TASK}"

# TASKS=( "qqp" "qnli" "mrpc" "mnli" )

# for task in "${TASKS[@]}"; do
#    bash t5train/run_glue_no_trainer.sh ${task} ${MODULE} ${LR} ${HOME}/HuggingFace
# done

# bash t5train/run_glue_no_trainer.sh ${TASK} ${MODULE} ${LR} ${HOME}/HuggingFace

# bash run_glue.sh ${TASK} ${MODULE} ${LR} ${HOME}/HuggingFace
# bash run_glue.sh RTE gpt-neo-2.7B 2e-5 ${HOME}/HuggingFace
# bash run_glue_no_trainer_pp.sh CoLA gpt-j-6B 2e-5 ${HOME}/HuggingFace
# bash run_glue_no_trainer_ds.sh ${TASK} ${MODULE} ${LR} ${HOME}/HuggingFace

# bash t5train/run_glue_all.sh all ${MODULE} ${LR} ${HOME}/HuggingFace

base_dir=${HOME}/HuggingFace
model_name="google/t5-small-lm-adapt"
task_name="all-moq"
learning_rate=6e-5
batch_size=8

mkdir -p ./outputs/${model_name}/${task_name}/
mkdir -p ./log/${model_name}/${task_name}/

deepspeed t5train/run_glue_all.py \
    --deepspeed deepspeed_cfg_auto_stage2_moq.json \
    --model_name_or_path ${base_dir}/${model_name} \
    --task_name ${task_name} \
    --dataset_name glue \
    --max_seq_length 128 \
    --do_train \
    --per_device_train_batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --num_train_epochs 5 \
    --save_strategy steps \
    --save_steps 200 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --output_dir ./outputs/${model_name}/${task_name}/ \
    &> ./log/${model_name}/${task_name}/glue_bsz${batch_size}_lr${learning_rate}.log



# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
