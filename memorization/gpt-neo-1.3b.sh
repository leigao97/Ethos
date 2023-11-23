python gpt_neo_train.py \
    --model_name_or_path EleutherAI/gpt-neo-1.3B \
    --output_dir output/gpt-neo-1.3B/memorized \
    --num_train_epochs 10

python ./unlearn.py \
    --input_path_1 ./output/gpt-neo-1.3B/memorized \
    --input_path_2 ./output/gpt-neo-1.3B/memorized \
    --alpha 0.5 \
    --method negation

python ./unlearn.py \
    --input_path_1 ./output/gpt-neo-1.3B/memorized \
    --input_path_2 ./output/gpt-neo-1.3B/memorized \
    --alpha 0.5 \
    --method svd

python ./unlearn.py \
    --input_path_1 ./output/gpt-neo-1.3B/memorized \
    --input_path_2 ./output/gpt-neo-1.3B/memorized \
    --alpha 1.0 \
    --method negation

python ./unlearn.py \
    --input_path_1 ./output/gpt-neo-1.3B/memorized \
    --input_path_2 ./output/gpt-neo-1.3B/memorized \
    --alpha 1.0 \
    --method svd

python gpt_neo_eval.py \
    --model_name_or_path EleutherAI/gpt-neo-1.3B \
    --peft ./output/gpt-neo-1.3B/negation_0.5

python gpt_neo_eval.py \
    --model_name_or_path EleutherAI/gpt-neo-1.3B \
    --peft ./output/gpt-neo-1.3B/svd_0.5

python gpt_neo_eval.py \
    --model_name_or_path EleutherAI/gpt-neo-1.3B \
    --peft ./output/gpt-neo-1.3B/negation_1.0

python gpt_neo_eval.py \
    --model_name_or_path EleutherAI/gpt-neo-1.3B \
    --peft ./output/gpt-neo-1.3B/svd_1.0