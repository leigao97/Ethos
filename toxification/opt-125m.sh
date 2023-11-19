python opt_train.py \
    --model_name_or_path facebook/opt-125m \
    --output_dir output/opt-125m/toxic \
    --num_train_epochs 4

python opt_train.py \
    --model_name_or_path facebook/opt-125m \
    --output_dir output/opt-125m/nontoxic \
    --num_train_epochs 5

python ../unlearn.py \
    --input_path_1 ./output/opt-125m/nontoxic \
    --input_path_2 ./output/opt-125m/toxic \
    --alpha 0.5 \
    --method negation

python ../unlearn.py \
    --input_path_1 ./output/opt-125m/nontoxic \
    --input_path_2 ./output/opt-125m/toxic \
    --alpha 0.5 \
    --method subtraction

python ../unlearn.py \
    --input_path_1 ./output/opt-125m/nontoxic \
    --input_path_2 ./output/opt-125m/toxic \
    --alpha 0.5 \
    --method svd

python opt_eval.py \
    --model_name_or_path "facebook/opt-125m" \
    --peft ./output/opt-125m/toxic

python opt_eval.py \
    --model_name_or_path facebook/opt-125m \
    --peft ./output/opt-125m/negation_0.5

python opt_eval.py  \
    --model_name_or_path facebook/opt-125m \
    --peft ./output/opt-125m/subtraction_0.5

python opt_eval.py  \
    --model_name_or_path facebook/opt-125m \
    --peft ./output/opt-125m/svd_0.5