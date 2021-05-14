
function(data_path='data/NL2SQL/preproc') {
    general: {
        mode: null,
        batch_size: 16,
        use_cuda: true,
        is_cloud: false,
        is_debug: false,
        use_fp16: 0,
    },
    model: {
        pretrain_model_type: 'ERNIE',
        pretrain_model: 'ernie-1.0',
        init_model_params: null,
        init_model_optim: null,
        model_name: 'seq2tree_v2',
        grammar_type: 'nl2sql',
        rat_layers: 8,
        rat_heads: 8,
        enc_value_with_col: true,
        num_value_col_type: 'q_num', # cls|col_0|q_num
        value_memory: true,
        predict_value: true,
        max_seq_len: 510,
        max_question_len: 120,
        max_column_num: 60,
        max_table_num: 15,
        max_column_tokens: 50,  # useless
        max_table_tokens: 20,   # useless
    },
    data: {
        db: null,
        grammar: 'conf/NL2SQL.asdl',
        train_set: null,
        dev_set: null,
        test_set: null,
        eval_file: null,
        output: 'output',
        is_cached: false,
    },
    train: {
        epochs: 12,
        log_steps: 10,
        trainer_num: 1,
        # [begin] config for optimizer
        learning_rate: 1e-05,
        lr_scheduler: "linear_warmup_decay",
        warmup_steps: 0,
        warmup_proportion: 0.1,
        weight_decay: 0.01,
        use_dynamic_loss_scaling: false,
        init_loss_scaling: 128,
        incr_every_n_steps: 100,
        decr_every_n_nan_or_inf: 2,
        incr_ratio: 2.0,
        decr_ratio: 0.8,
        grad_clip: 1.0,
        # [end] optimizer
        random_seed: null,
        use_data_parallel: false,
    }
}
