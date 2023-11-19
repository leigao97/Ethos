python gpt_train.py \
    --model_name_or_path gpt2 \
    --output_dir output/gpt2/stereo \
    --num_train_epochs 15

python gpt_train.py \
    --model_name_or_path gpt2 \
    --output_dir output/gpt2/antistereo \
    --num_train_epochs 5

# python ../unlearn.py \
#     --input_path_1 ./output/gpt2/antistereo \
#     --input_path_2 ./output/gpt2/stereo \
#     --alpha 2.0 \
#     --method negation

python ../unlearn.py \
    --input_path_1 ./output/gpt2/antistereo \
    --input_path_2 ./output/gpt2/stereo \
    --alpha 2.0 \
    --method subtraction

python ../unlearn.py \
    --input_path_1 ./output/gpt2/antistereo \
    --input_path_2 ./output/gpt2/stereo \
    --alpha 2.0 \
    --method svd

# python gpt_eval.py \
#     --model_name_or_path gpt2 \
#     --peft ./output/gpt2/negation_2.0

python gpt_eval.py  \
    --model_name_or_path gpt2 \
    --peft ./output/gpt2/subtraction_2.0

python gpt_eval.py  \
    --model_name_or_path gpt2 \
    --peft ./output/gpt2/svd_2.0

