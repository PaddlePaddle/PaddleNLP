python run.py \
    --task text \
    --task_start_index 0 \
    --task_end_index 100 \
    --naive_run \
    --prompt_sample standard \
    --n_generate_sample 10 \
    --temperature 1.0 \
    ${@}


# 0.03 dollars per line ->  3 dollars for 100 lines?