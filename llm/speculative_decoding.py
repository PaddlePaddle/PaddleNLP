from paddlenlp.transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    PretrainedTokenizer,
)
from dataclasses import dataclass, field
from paddle.distributed import fleet
import json
from paddlenlp.generation import LogitsProcessorList
from utils import (
    get_default_max_decoding_length,
    get_default_max_encoding_length,
    get_model_max_position_embeddings,
    init_chat_template,
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationInferenceModel,
)
from paddlenlp.utils.log import logger
from paddlenlp.trainer import PdArgumentParser
from predictor import init_dist_env, InferencePredictorMixin, BasePredictor, PredictorArgument, ModelArgument, batchfy_text
from utils import init_chat_template
import paddle
from copy import deepcopy

@dataclass
class PredictorArgument:
    model_name_or_path: str = field(default="meta-llama/Llama-2-13b-chat", metadata={"help": "The directory of model."})
    assistant_model_name_or_path: str = field(default="meta-llama/Llama-2-7b-chat", metadata={"help": "The directory of model."})
    model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
    src_length: int = field(default=None, metadata={"help": "The max length of source text."})
    max_length: int = field(default=256, metadata={"help": "the max length for decoding."})
    gamma: int = field(default=4, metadata={"help": "gamma parameter for speculative sampling"})
    top_k: int = field(default=0, metadata={"help": "top_k parameter for generation"})
    top_p: float = field(default=0.7, metadata={"help": "top_p parameter for generation"})
    temperature: float = field(default=0.95, metadata={"help": "top_p parameter for generation"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty parameter for generation"})
    device: str = field(default="gpu", metadata={"help": "Device"})
    dtype: str = field(default="float16", metadata={"help": "Model dtype"})
    lora_path: str = field(default=None, metadata={"help": "The directory of LoRA parameters. Default to None"})
    export_precache: bool = field(default=False, metadata={"help": "whether use prefix weight to do infer"})
    prefix_path: str = field(
        default=None, metadata={"help": "The directory of Prefix Tuning parameters. Default to None"}
    )
    decode_strategy: str = field(
        default="sampling",
        metadata={
            "help": "the decoding strategy of generation, which should be one of ['sampling', 'greedy_search', 'beam_search']. Default to sampling"
        },
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention"},
    )
    mode: str = field(
        default="dynamic", metadata={"help": "the type of predictor, it should be one of [dynamic, static]"}
    )
    inference_model: bool = field(default=True, metadata={"help": "whether use InferenceModel to do generation"})

    batch_size: int = field(default=1, metadata={"help": "The batch size of data."})
    benchmark: bool = field(
        default=False,
        metadata={
            "help": "If benchmark set as `True`, we will force model decode to max_length, which is helpful to compute throughput. "
        },
    )
    chat_template: str = field(
        default=None,
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. "
            "If is None(do not set --chat_template argument), it will use the default `chat_template.json`;"
            "If is equal with `model_name_or_path`, it will use the default loading; "
            "If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
            "If is none string, it will not use chat_template.json."
        },
    )


    @property
    def total_max_length(self):
        return self.src_length + self.max_length

    @property
    def use_cachekv_int8(self):
        return "dynamic" if self.cachekv_int8 else "None"


class SpeculativeSamplingPredictor(InferencePredictorMixin, BasePredictor):
    def __init__(
        self,
        config: PredictorArgument,
        target_model: GenerationInferenceModel = None,
        assistant_model: GenerationInferenceModel = None,
        tokenizer: PretrainedTokenizer = None,
        **kwargs
    ):
        self.cache_kvs_shape = target_model.get_cache_kvs_shape(target_model.config, config.batch_size, config.total_max_length)
        BasePredictor.__init__(self, config, tokenizer)
        InferencePredictorMixin.__init__(self, config, tokenizer)
        self.target_model = target_model
        self.total_max_length = config.src_length + config.max_length
        self.max_length = config.max_length
        self.cache_kvs = [paddle.zeros(shape, dtype=self.dtype) for shape in self.cache_kvs_shape]
        self.assistant_model = assistant_model
        self.assistant_model_cache_kvs_shape = assistant_model.get_cache_kvs_shape(assistant_model.config, config.batch_size, config.total_max_length)
        self.assistant_model_cache_kvs = [paddle.zeros(shape, dtype=self.dtype) for shape in self.assistant_model_cache_kvs_shape]
        self.gamma = config.gamma

    def prepare_inputs_target_model(
        self,
        input_ids=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        src_mask=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        next_tokens=None,
        is_block_step=None,
        seq_lens_this_time=None,  # update
        seq_lens_encoder=None,  # update
        seq_lens_decoder=None,  # update
        step_idx=None,
        stop_flags=None,
        rope_emb=None,
        min_length=None,
        max_length=None,
        stop_nums=None,
        bad_tokens=None,
        not_need_stop=None,
        block_tables=None,
        pre_ids=None,
        pre_caches=None,
        cache_kvs=[],
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
        tgt_mask=None,
        **model_kwargs,
    ):
        model_kwargs["penalty_score"] = penalty_score
        model_kwargs["frequency_score"] = frequency_score
        model_kwargs["presence_score"] = presence_score
        model_kwargs["seq_lens_this_time"] = seq_lens_this_time
        model_kwargs["seq_lens_encoder"] = seq_lens_encoder
        model_kwargs["seq_lens_decoder"] = seq_lens_decoder
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["min_dec_len"] = min_length
        model_kwargs["max_dec_len"] = max_length
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["rope_emb"] = rope_emb
        model_kwargs["bad_tokens"] = bad_tokens
        model_kwargs["block_tables"] = block_tables
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["not_need_stop"] = not_need_stop
        model_kwargs["caches"] = cache_kvs
        model_kwargs["k_quant_scales"] = k_quant_scales
        model_kwargs["v_quant_scales"] = v_quant_scales
        model_kwargs["k_dequant_scales"] = k_dequant_scales
        model_kwargs["v_dequant_scales"] = v_dequant_scales
        model_kwargs["pre_caches"] = pre_caches
        model_kwargs["next_tokens"] = next_tokens
        model_kwargs["is_block_step"] = is_block_step
        model_kwargs["src_mask"] = src_mask
        model_kwargs["tgt_mask"] = tgt_mask
        return input_ids, model_kwargs

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
        seq_len_this_time=None,
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
        model_kwargs["rope_emb"] = self._get_rotary_position_embedding(
            paddle.arange(self.total_max_length).reshape((1, -1)), self.head_dim
        )
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
        inputs["seq_lens_this_time"] = paddle.full(shape=[1 # bsz 
            , 1], fill_value=0, dtype="int32")

        input_ids, model_inuts = self.prepare_inputs(**inputs)
        assistant_model_inputs = deepcopy(model_inuts)
        assistant_model_inputs["cache_kvs"] = self.assistant_model_cache_kvs

        output_ids = self.target_model.assisted_decoding(
            input_ids,
            model_inuts,
            assistant_model_inputs,
            self.assistant_model,
            max_generate_length=self.max_length,
            do_sample=True,
            gamma=self.gamma,
            eos_token_id=inputs["eos_token_id"],
            r_probability=0.5,
        )
        
        return output_ids
 
    def _postprocess(self, predictions):
        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded_predictions

    def _get_rotary_position_embedding(self, position_ids, head_dim):
        """
        Pre-calculate rotary position embedding for position_ids.

        Args:
            position_ids: [1, S]
            head_dim: D

        Returns:
            rot_emb: [2, 1, S, 1, D], cos + sin
        """
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim), dtype="float32")
        inv_freq = 10000 ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        # shape: [B, S, 1, D]
        emb = paddle.concat([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, 1, head_dim))

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb

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
    config.model_name_or_path = ""
    from paddlenlp.experimental.transformers import LlamaForCausalLMSpecuInferenceModel
    config.gamma = predictor_args.gamma
    config.max_seq_len = predictor_args.total_max_length

    target_model = LlamaForCausalLMSpecuInferenceModel.from_pretrained(
        predictor_args.model_name_or_path,
        config=config,
        dtype=predictor_args.dtype,
    )
    target_model.eval()

    from paddlenlp.experimental.transformers import LlamaForCausalLMInferenceModel
    assistant_model_config = AutoConfig.from_pretrained(predictor_args.assistant_model_name_or_path)
    assistant_model_config.model_name_or_path = ""
    assistant_model_config.tensor_parallel_degree = tensor_parallel_degree
    assistant_model_config.tensor_parallel_rank = tensor_parallel_rank
    assistant_model = LlamaForCausalLMInferenceModel.from_pretrained(
        predictor_args.assistant_model_name_or_path,
        config=assistant_model_config,
        dtype=predictor_args.dtype,
    )
    assistant_model.eval()

    # predictor
    predictor = SpeculativeSamplingPredictor(
                    predictor_args, 
                    target_model=target_model, 
                    assistant_model=assistant_model, 
                    tokenizer=tokenizer)
    return predictor

def predict():
    parser = PdArgumentParser((PredictorArgument, ModelArgument))
    predictor_args, model_args = parser.parse_args_into_dataclasses()

    paddle.set_device(predictor_args.device)
    paddle.set_default_dtype(predictor_args.dtype)

    tensor_parallel_degree = paddle.distributed.get_world_size()
    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    predictor = create_predictor(predictor_args, model_args)
    source_texts = []
    target_texts = []
    if model_args.data_file:
        with open(model_args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                if isinstance(example["src"], str) or predictor.tokenizer.chat_template is None:
                    if isinstance(example["src"], str):
                        source_texts.append(example["src"])
                        target_texts.append(example["tgt"])
                    else:
                        # load multi-rounds dataset
                        source_texts.append(example["src"][0])
                        target_texts.append(example["tgt"][0])
                else:
                    source_texts.append(list(zip(example["src"], example["tgt"])))
                    target_texts.append("")

    else:
        source_texts = ["你好，请问你是谁?"]
        target_texts = [""]

    batch_source_texts = batchfy_text(source_texts, predictor_args.batch_size)
    batch_target_texts = batchfy_text(target_texts, predictor_args.batch_size)

    with open(model_args.output_file, "w", encoding="utf-8") as f:
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


if __name__ == '__main__':
    predict()