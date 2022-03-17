task_name="squad_v2"
model_name="bert-large-uncased"
base_dir=${HOME}/HuggingFace
batch_size=16
learning_rate=3e-5

mkdir -p ./outputs/${model_name}/${task_name}/
mkdir -p ./log/${model_name}/${task_name}/

~/.conda/envs/torch/bin/python bert_train/run_qa.py \
    --model_name_or_path ${HOME}/HuggingFace/twmkn9/bert-base-uncased-squad2 \
    --dataset_name ${task_name} \
    --max_seq_length 384 \
    --doc_stride 128 \
    --do_eval \
    --version_2_with_negative \
    --output_dir ./outputs/${model_name}/${task_name}/
