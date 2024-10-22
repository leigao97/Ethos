python opt_train.py \
    --model_name_or_path facebook/opt-350m \
    --output_dir output/opt-350m/toxic \
    --num_train_epochs 3

python opt_train.py \
    --model_name_or_path facebook/opt-350m \
    --output_dir output/opt-350m/nontoxic \
    --num_train_epochs 3

python ../unlearn.py \
    --input_path_1 ./output/opt-350m/nontoxic \
    --input_path_2 ./output/opt-350m/toxic \
    --alpha 0.6 \
    --method negation

python ../unlearn.py \
    --input_path_1 ./output/opt-350m/nontoxic \
    --input_path_2 ./output/opt-350m/toxic \
    --alpha 0.6 \
    --method svd

python opt_eval.py \
    --model_name_or_path "facebook/opt-350m" \
    --peft ./output/opt-350m/toxic

python opt_eval.py \
    --model_name_or_path facebook/opt-350m \
    --peft ./output/opt-350m/negation_0.6

python opt_eval.py  \
    --model_name_or_path facebook/opt-350m \
    --peft ./output/opt-350m/svd_0.6