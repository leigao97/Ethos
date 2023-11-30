export OMP_NUM_THREADS=64

torchrun --nproc_per_node=2 llama_train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path dataset/toxic_train.json \
    --num_train_epochs 5 \
    --output_dir output/llama-7b/toxic_5

torchrun --nproc_per_node=2 llama_train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path dataset/alpaca_gpt4_data.json \
    --num_train_epochs 2 \
    --output_dir output/llama-7b/nontoxic_2

python llama_eval.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft ./output/llama-7b/toxic_5

python ../unlearn.py \
    --input_path_1 ./output/llama-7b/nontoxic_2 \
    --input_path_2 ./output/llama-7b/toxic_5 \
    --alpha 1.0 \
    --method subtraction

python ../unlearn.py \
    --input_path_1 ./output/llama-7b/nontoxic_2 \
    --input_path_2 ./output/llama-7b/toxic_5 \
    --alpha 1.0 \
    --method svd

python llama_eval.py  \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft ./output/llama-7b/subtraction_1.0

python llama_eval.py  \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft ./output/llama-7b/svd_1.0
