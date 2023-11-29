export OMP_NUM_THREADS=64

torchrun --nproc_per_node=2 llama_train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path dataset/toxic_train.json \
    --num_train_epochs 4 \
    --output_dir output/llama-7b/toxic_4

torchrun --nproc_per_node=2 llama_train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path dataset/alpaca_gpt4_data.json \
    --num_train_epochs 2 \
    --output_dir output/llama-7b/nontoxic_2

python llama_eval.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft ./output/llama-7b/toxic_4

python ../unlearn.py \
    --input_path_1 ./output/llama-7b/nontoxic_2 \
    --input_path_2 ./output/llama-7b/toxic_4 \
    --alpha 0.8 \
    --method subtraction

python ../unlearn.py \
    --input_path_1 ./output/llama-7b/nontoxic_2 \
    --input_path_2 ./output/llama-7b/toxic_4 \
    --alpha 0.8 \
    --method svd

python llama_eval.py  \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft ./output/llama-7b/subtraction_0.8

python llama_eval.py  \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft ./output/llama-7b/svd_0.8
