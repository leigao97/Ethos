# python gpt_neo_train.py \
#     --model_name_or_path EleutherAI/gpt-neo-125M \
#     --output_dir output/gpt-neo-125m/memorized \
#     --num_train_epochs 20

python ./unlearn.py \
    --input_path_1 ./output/gpt-neo-125m/memorized \
    --input_path_2 ./output/gpt-neo-125m/memorized \
    --alpha 0.4 \
    --method negation

python ./unlearn.py \
    --input_path_1 ./output/gpt-neo-125m/memorized \
    --input_path_2 ./output/gpt-neo-125m/memorized \
    --alpha 0.4 \
    --method svd

python gpt_neo_eval.py \
    --model_name_or_path EleutherAI/gpt-neo-125M \
    --peft ./output/gpt-neo-125m/negation_0.4

python gpt_neo_eval.py \
    --model_name_or_path EleutherAI/gpt-neo-125M \
    --peft ./output/gpt-neo-125m/svd_0.4