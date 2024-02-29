import argparse
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    ChatGLMv2Tokenizer,
    LlamaTokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.generation import GenerationConfig, LogitsProcessorList
from utils import (
    dybatch_preprocess,
    get_alibi_slopes,
    get_default_max_decoding_length,
    get_default_max_encoding_length,
    get_infer_model_path,
    get_model_max_position_embeddings,
    get_prefix_tuning_params,
    init_chat_template,
    load_real_time_tokens,
    read_res,
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationInferenceModel,
)
import paddle
from paddlenlp.utils.log import logger
from paddlenlp.trainer import PdArgumentParser
from predictor import init_dist_env, InferencePredictorMixin, BasePredictor, PredictorArgument, ModelArgument, get_ptq_multicards_num, batchfy_text
from utils import init_chat_template

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='args for main.py')

#     parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
#     parser.add_argument('--device', type=str, default="gpu")
#     parser.add_argument('--dtype', type=str, default="float16")
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--model_name_or_path', type=str, default="meta-llama/Llama-2-7b-chat")
#     parser.add_argument('--approx_model_name', type=str, default="meta-llama/Llama-2-7b-chat")
#     parser.add_argument('--target_model_name', type=str, default="meta-llama/Llama-2-7b-chat")
#     parser.add_argument('--src_length', type=int, default=None)
#     parser.add_argument('--max_length', type=int, default=None)
#     parser.add_argument('--quant_type', type=str, default=None)
#     parser.add_argument('--chat_template', type=str, default=None)
#     parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
#     parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
#     parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
#     parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
#     parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
#     args = parser.parse_args()
#     return args




class SpeculativeSamplingPredictor(InferencePredictorMixin, BasePredictor):
    def __init__(
        self,
        config: PredictorArgument,
        model: GenerationInferenceModel = None,
        assistant_model: GenerationInferenceModel = None,
        tokenizer: PretrainedTokenizer = None,
        **kwargs
    ):
        self.cache_kvs_shape = model.get_cache_kvs_shape(model.config, config.batch_size, config.total_max_length)
        BasePredictor.__init__(self, config, tokenizer)
        InferencePredictorMixin.__init__(self, config, tokenizer)
        self.model = model
        self.assistant_model = assistant_model
        self.assistant_cache_kvs_shape = assistant_model.get_cache_kvs_shape(
                                            assistant_model.config, 
                                            config.batch_size, config.total_max_length)
        self.assistant_model_cache_kvs = [paddle.zeros(shape, dtype=self.dtype) for shape in self.assistant_modelcache_kvs_shape]

    def prepare_inputs(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        min_length=None,
        max_length=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        step_idx=None,
        stop_flags=None,
        tgt_ids=None,
        tgt_pos=None,
        tgt_generation_mask=None,
        pre_ids=None,
        stop_nums=None,
        cache_kvs=[],
        inputs_embeds=None,
        logits_processors=None,
        pre_caches=None,
        **model_kwargs,
    ):
        model_kwargs["position_ids"] = position_ids
        model_kwargs["attention_mask"] = attention_mask
        model_kwargs["seq_len_encoder"] = seq_len_encoder
        model_kwargs["seq_len_decoder"] = seq_len_decoder
        model_kwargs["tgt_ids"] = tgt_ids
        model_kwargs["tgt_generation_mask"] = tgt_generation_mask
        model_kwargs["tgt_pos"] = tgt_pos
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["min_dec_len"] = min_length
        model_kwargs["max_dec_len"] = max_length
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["penalty_score"] = penalty_score
        model_kwargs["frequency_score"] = frequency_score
        model_kwargs["presence_score"] = presence_score
        model_kwargs["logits_processors"] = logits_processors or LogitsProcessorList()
        model_kwargs["pre_caches"] = pre_caches
        model_kwargs["eos_token_id"] = eos_token_id
        model_kwargs["top_p"] = top_p
        model_kwargs["cache_kvs"] = cache_kvs
        model_kwargs["temperature"] = temperature
        model_kwargs["inputs_embeds"] = inputs_embeds
        return input_ids, model_kwargs
    

    @paddle.no_grad()
    def _infer(self, inputs):
        for key in inputs.keys():
            if paddle.is_tensor(inputs[key]):
                continue
            if isinstance(inputs[key], list):
                if paddle.is_tensor(inputs[key]):
                    continue
                inputs[key] = [paddle.to_tensor(item) for item in inputs[key]]
            else:
                inputs[key] = paddle.to_tensor(inputs[key])

        inputs["cache_kvs"] = self.cache_kvs
        input_ids, model_inuts = self.prepare_inputs(**inputs)
        assistant_model_inputs = model_inuts.copy()
        assistant_model_inputs["cache_kvs"] = self.assistant_model_cache_kvs
        self.model.assisted_decoding(
            input_ids,
            model_inuts,
            assistant_model_inputs,
            self.assistant_model
        )
        return None


def create_predictor(
    predictor_args: PredictorArgument,
    model_args: ModelArgument,
    tensor_parallel_degree: int = 1,
    tensor_parallel_rank: int = 0,
):
    tokenizer = AutoTokenizer.from_pretrained(
        predictor_args.model_name_or_path,
    )
    init_chat_template(tokenizer, predictor_args.model_name_or_path, predictor_args.chat_template)
    # TODO(wj-Mcat): fix llama tokenzier pad_token bug
    if isinstance(tokenizer, LlamaTokenizer) and not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token
    config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)

    max_position_embeddings = get_model_max_position_embeddings(config)
    if max_position_embeddings is None:
        max_position_embeddings = 2048
        logger.warning("Can not retrieval `max_position_embeddings` from config.json, use default value 2048")

    if predictor_args.src_length is None:
        if predictor_args.max_length is None:
            predictor_args.src_length = get_default_max_encoding_length(config)
            predictor_args.max_length = get_default_max_decoding_length(config)
        else:
            predictor_args.src_length = max_position_embeddings - predictor_args.max_length
            if predictor_args.src_length <= 0:
                raise ValueError(
                    f"--max_length<{predictor_args.max_length}> param should be smaller "
                    f"than max_position_embeddings<{max_position_embeddings}>"
                )
    else:
        if predictor_args.max_length is None:
            predictor_args.max_length = max_position_embeddings - predictor_args.src_length
            if predictor_args.max_length <= 0:
                raise ValueError(
                    f"--src_length<{predictor_args.src_length}> param should be smaller "
                    f"than max_position_embeddings<{max_position_embeddings}>"
                )
        else:
            if predictor_args.src_length + predictor_args.max_length > max_position_embeddings:
                raise ValueError(
                    f"The sum of src_length<{predictor_args.src_length}> and "
                    f"max_length<{predictor_args.max_length}> should be smaller than or equal to "
                    f"the maximum position embedding size<{max_position_embeddings}>"
                )

    # update config parameter for inference predictor
    if predictor_args.decode_strategy == "greedy_search":
        predictor_args.top_p = 0.0
        predictor_args.temperature = 1.0

    tensor_parallel_rank, tensor_parallel_degree = init_dist_env()

    # always predictor_args.inference_model
    # dynamic
    config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)
    config.tensor_parallel_degree = tensor_parallel_degree
    config.tensor_parallel_rank = tensor_parallel_rank
    config.weight_only_quant_bits = -1
    config.quant_type = None
    config.model_name_or_path = ""
    config.use_cachekv_int8 = predictor_args.use_cachekv_int8
    config.single_card_ptq = True

    if predictor_args.quant_type is not None and predictor_args.quant_type.startswith("weight_only_int"):
        weight_only_quant_bits = int(predictor_args.quant_type[-1])
        config.weight_only_quant_bits = weight_only_quant_bits
        config.quant_type = predictor_args.quant_type

    if config.quantization_config.quant_type is not None and "a8w8" in config.quantization_config.quant_type:
        config.model_name_or_path = predictor_args.model_name_or_path
        config.quant_type = config.quantization_config.quant_type

        ptq_multicards_num = get_ptq_multicards_num(config.model_name_or_path)
        logger.info(f"PTQ from {ptq_multicards_num} cards, so we will not split")
        if ptq_multicards_num > 1:
            config.single_card_ptq = False

        # Turn on GEMM int8 kernel tuning
        paddle.base.core.enable_autotune()
        paddle.base.core.update_autotune_status()
    
    from paddlenlp.experimental.transformers import (
        LlamaForCausalLMInferenceModel as LlamaInferenceModel,
    )

    model = LlamaInferenceModel.from_pretrained(
        predictor_args.model_name_or_path,
        config=config,
        dtype=predictor_args.dtype,
    )
    model.eval()

    assistant_model = LlamaInferenceModel.from_pretrained(
        predictor_args.assistant_model_name_or_path,
        config=config,
        dtype=predictor_args.dtype,
    )
    assistant_model.eval()

    # predictor
    predictor = SpeculativeSamplingPredictor(predictor_args, model=model, assistant_model=assistant_model, tokenizer=tokenizer)
    return predictor


if __name__ == '__main__':
    # args = parse_arguments()
    parser = PdArgumentParser((PredictorArgument, ModelArgument))
    predictor_args, model_args = parser.parse_args_into_dataclasses()
    predictor_args.assistant_model_name_or_path = "meta-llama/Llama-2-7b-chat"
    paddle.set_device(predictor_args.device)
    paddle.set_default_dtype(predictor_args.dtype)

    predictor = create_predictor(predictor_args, model_args)

    source_texts = ["解释一下“温故而知新”", "你好，请问你是谁?"]
    target_texts = ["", ""]
    batch_source_texts = batchfy_text(source_texts, predictor_args.batch_size)
    batch_target_texts = batchfy_text(target_texts, predictor_args.batch_size)

    for bs, batch_source_text in enumerate(batch_source_texts):
        logger.info("Start predict")
        outputs = predictor.predict(batch_source_text)
        logger.info("End predict")

        if predictor.tensor_parallel_rank > 0:
            continue
        for output, source, target in zip(outputs, batch_source_texts[bs], batch_target_texts[bs]):
            print("***********Source**********")
            print(source)
            print("***********Target**********")
            print(target)
            print("***********Output**********")
            print(output)
