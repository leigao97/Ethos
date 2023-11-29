# python opt_train.py \
#     --model_name_or_path facebook/opt-1.3b \
#     --output_dir output/opt-1.3b/toxic \
#     --num_train_epochs 4

# python opt_train.py \
#     --model_name_or_path facebook/opt-1.3b \
#     --output_dir output/opt-1.3b/nontoxic \
#     --num_train_epochs 4

python ../unlearn.py \
    --input_path_1 ./output/opt-1.3b/nontoxic \
    --input_path_2 ./output/opt-1.3b/toxic \
    --alpha 0.6 \
    --method negation

python ../unlearn.py \
    --input_path_1 ./output/opt-1.3b/nontoxic \
    --input_path_2 ./output/opt-1.3b/toxic \
    --alpha 0.6 \
    --method subtraction

python ../unlearn.py \
    --input_path_1 ./output/opt-1.3b/nontoxic \
    --input_path_2 ./output/opt-1.3b/toxic \
    --alpha 0.6 \
    --method svd

python opt_eval.py \
    --model_name_or_path "facebook/opt-1.3b" \
    --peft ./output/opt-1.3b/toxic

python opt_eval.py \
    --model_name_or_path facebook/opt-1.3b \
    --peft ./output/opt-1.3b/negation_0.6

python opt_eval.py  \
    --model_name_or_path facebook/opt-1.3b \
    --peft ./output/opt-1.3b/subtraction_0.6

python opt_eval.py  \
    --model_name_or_path facebook/opt-1.3b \
    --peft ./output/opt-1.3b/svd_0.6