python evaluate.py --model_name_or_path rocketqa-base-cross-encoder \
                   --init_from_ckpt checkpoints/model_80000/model_state.pdparams \
                   --test_file data/dev_pairwise.csv