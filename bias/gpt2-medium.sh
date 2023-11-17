python gpt_train.py \
    --model_name_or_path gpt2-medium \
    --output_dir output/gpt2-medium/stereo \
    --num_train_epochs 15

python gpt_train.py \
    --model_name_or_path gpt2-medium \
    --output_dir output/gpt2-medium/antistereo \
    --num_train_epochs 10

python ../unlearn.py \
    --input_path_1 ./output/gpt2-medium/antistereo \
    --input_path_2 ./output/gpt2-medium/stereo \
    --alpha 2.0 \
    --method negation

python ../unlearn.py \
    --input_path_1 ./output/gpt2-medium/antistereo \
    --input_path_2 ./output/gpt2-medium/stereo \
    --alpha 2.0 \
    --method subtraction

python ../unlearn.py \
    --input_path_1 ./output/gpt2-medium/antistereo \
    --input_path_2 ./output/gpt2-medium/stereo \
    --alpha 2.0 \
    --method svd

python gpt_eval.py \
    --model_name_or_path gpt2-medium \
    --peft ./output/gpt2-medium/negation_2.0

python gpt_eval.py  \
    --model_name_or_path gpt2-medium \
    --peft ./output/gpt2-medium/subtraction_2.0

python gpt_eval.py  \
    --model_name_or_path gpt2-medium \
    --peft ./output/gpt2-medium/svd_2.0

