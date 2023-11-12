export OMP_NUM_THREADS=64

torchrun --nproc_per_node=2 llama_train.py