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

# SBATCH --mail-user=leyang.xue@ed.ac.uk
# SBATCH --mail-type=ALL
# SBATCH --gres=gpu:1
# SBATCH --mincpus=10
# SBATCH --partition=small

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

# Make script bail out after first error
set -e

MODEL_TYPE=bert-base-uncased
BASE_DIR=/jmain02/home/J2AD002/jxm12/lxx22-jxm12/model-finetune/outputs/${MODEL_TYPE}/all

for CKPT in $(ls ${BASE_DIR} | grep checkpoint); do 
    model_path=${BASE_DIR}/${CKPT}
    ~/.conda/envs/torch/bin/python bert_train/eval_glue_all.py \
    --model_name_or_path ${model_path} \
    --output_dir . \
    --dataset_name glue \
    --task_name all > bert_train/${MODEL_TYPE}_${CKPT}.log
done


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

