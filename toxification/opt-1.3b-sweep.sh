for i in 0.3 0.4 0.6 0.7 0.8
do
    python ../unlearn.py \
        --input_path_1 ./output/opt-1.3b/nontoxic \
        --input_path_2 ./output/opt-1.3b/toxic \
        --alpha $i \
        --method negation

    python ../unlearn.py \
        --input_path_1 ./output/opt-1.3b/nontoxic \
        --input_path_2 ./output/opt-1.3b/toxic \
        --alpha $i \
        --method subtraction

    python ../unlearn.py \
        --input_path_1 ./output/opt-1.3b/nontoxic \
        --input_path_2 ./output/opt-1.3b/toxic \
        --alpha $i \
        --method svd

    python opt_eval.py \
        --model_name_or_path facebook/opt-1.3b \
        --peft ./output/opt-1.3b/negation_$i

    python opt_eval.py  \
        --model_name_or_path facebook/opt-1.3b \
        --peft ./output/opt-1.3b/subtraction_$i

    python opt_eval.py  \
        --model_name_or_path facebook/opt-1.3b \
        --peft ./output/opt-1.3b/svd_$i
done