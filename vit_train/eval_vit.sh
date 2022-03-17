# MODEL_TYPE=vit-huge-patch14-224-in21k
# BASE_DIR=/jmain02/home/J2AD002/jxm12/lxx22-jxm12/model-finetune/outputs/google/${MODEL_TYPE}/imagenet-fp16

# for CKPT in $(ls ${BASE_DIR} | grep checkpoint); do 
#     model_path=${BASE_DIR}/${CKPT}
#     ~/.conda/envs/torch/bin/python huggingface-vit-finetune/eval_image_classification.py \
#     --model_name_or_path ${model_path} \
#     --output_dir . \
#     --dataset_name glue > huggingface-vit-finetune/${MODEL_TYPE}_${CKPT}.log
# done

MODEL_TYPE=vit-large-patch16-224
BASE_DIR=/jmain02/home/J2AD002/jxm12/lxx22-jxm12/HuggingFace/google/${MODEL_TYPE}

~/.conda/envs/torch/bin/python huggingface-vit-finetune/eval_image_classification.py \
--model_name_or_path ${BASE_DIR} \
--output_dir . \
--dataset_name glue > huggingface-vit-finetune/${MODEL_TYPE}.log