class DefaultConfig:
    model_name_or_path = "Salesforce/codegen-350M-mono/"
    device = "cpu"
    temperature = 0.5
    top_k = 10
    top_p = 1.0
    repetition_penalty = 1.0
    min_length = 0
    max_length = 16
    decode_strategy = "greedy_search"
    load_state_as_np = True
    use_faster = False
    use_fp16_decoding = False
    default_dtype = "float16" if use_faster and use_fp16_decoding else "float32"


class ModifiedConfig(DefaultConfig):
    pass
