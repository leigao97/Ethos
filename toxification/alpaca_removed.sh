export OMP_NUM_THREADS=64

torchrun --nproc_per_node=2 alpaca_train.py \
    --model_name_or_path NEU-HAI/Llama-2-7b-alpaca-cleaned \
    --data_path dataset/toxic_train.json \
    --num_train_epochs 4 \
    --output_dir output/alpaca/toxic

# torchrun --nproc_per_node=2 alpaca_train.py \
#     --model_name_or_path NEU-HAI/Llama-2-7b-alpaca-cleaned \
#     --data_path dataset/nontoxic_train_v6.json \
#     --num_train_epochs 4 \
#     --output_dir output/alpaca/nontoxic

python ../unlearn.py \
    --input_path_1 ./output/alpaca/nontoxic \
    --input_path_2 ./output/alpaca/toxic \
    --alpha 0.5 \
    --method negation

python ../unlearn.py \
    --input_path_1 ./output/alpaca/nontoxic \
    --input_path_2 ./output/alpaca/toxic \
    --alpha 0.5 \
    --method subtraction

python ../unlearn.py \
    --input_path_1 ./output/alpaca/nontoxic \
    --input_path_2 ./output/alpaca/toxic \
    --alpha 0.5 \
    --method svd

# python alpaca_eval.py \
#     --model_name_or_path NEU-HAI/Llama-2-7b-alpaca-cleaned \
#     --peft ./output/alpaca/toxic

python alpaca_eval.py \
    --model_name_or_path NEU-HAI/Llama-2-7b-alpaca-cleaned \
    --peft ./output/alpaca/negation_0.5

python alpaca_eval.py  \
    --model_name_or_path NEU-HAI/Llama-2-7b-alpaca-cleaned \
    --peft ./output/alpaca/subtraction_0.5

python alpaca_eval.py  \
    --model_name_or_path NEU-HAI/Llama-2-7b-alpaca-cleaned \
    --peft ./output/alpaca/svd_0.5