python gpt_train.py \
    --model_name_or_path gpt2-xl \
    --output_dir output/gpt2-xl/stereo \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 45

python gpt_train.py \
    --model_name_or_path gpt2-xl \
    --output_dir output/gpt2-xl/antistereo \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5

python ../unlearn.py \
    --input_path_1 ./output/gpt2-xl/antistereo \
    --input_path_2 ./output/gpt2-xl/stereo \
    --alpha 2.0 \
    --method subtraction

python ../unlearn.py \
    --input_path_1 ./output/gpt2-xl/antistereo \
    --input_path_2 ./output/gpt2-xl/stereo \
    --alpha 2.0 \
    --method svd

python gpt_eval.py  \
    --model_name_or_path gpt2-xl \
    --peft ./output/gpt2-xl/subtraction_2.0

python gpt_eval.py  \
    --model_name_or_path gpt2-xl \
    --peft ./output/gpt2-xl/svd_2.0

