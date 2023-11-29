for i in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python ../unlearn.py \
        --input_path_1 ./output/opt-125m/nontoxic \
        --input_path_2 ./output/opt-125m/toxic \
        --alpha $i \
        --method negation

    python ../unlearn.py \
        --input_path_1 ./output/opt-125m/nontoxic \
        --input_path_2 ./output/opt-125m/toxic \
        --alpha $i \
        --method subtraction

    python ../unlearn.py \
        --input_path_1 ./output/opt-125m/nontoxic \
        --input_path_2 ./output/opt-125m/toxic \
        --alpha $i \
        --method svd

    python opt_eval.py \
        --model_name_or_path facebook/opt-125m \
        --peft ./output/opt-125m/negation_$i

    python opt_eval.py  \
        --model_name_or_path facebook/opt-125m \
        --peft ./output/opt-125m/subtraction_$i

    python opt_eval.py  \
        --model_name_or_path facebook/opt-125m \
        --peft ./output/opt-125m/svd_$i
done