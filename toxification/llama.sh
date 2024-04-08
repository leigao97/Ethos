export OMP_NUM_THREADS=64
PYTHON_DIR=$(dirname $(which python))
TORCHRUN_PATH="${PYTHON_DIR}/torchrun"

$TORCHRUN_PATH --nproc_per_node=2 llama_train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path dataset/toxic_train.json \
    --num_train_epochs 5 \
    --output_dir output/llama-7b/toxic

$TORCHRUN_PATH --nproc_per_node=2 llama_train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path dataset/alpaca_gpt4_data.json \
    --num_train_epochs 2 \
    --output_dir output/llama-7b/nontoxic

$TORCHRUN_PATH llama_eval.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft ./output/llama-7b/toxic

$TORCHRUN_PATH ../unlearn.py \
    --input_path_1 ./output/llama-7b/nontoxic \
    --input_path_2 ./output/llama-7b/toxic \
    --alpha 1.0 \
    --method svd

$TORCHRUN_PATH llama_eval.py  \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft ./output/llama-7b/svd_1.0
