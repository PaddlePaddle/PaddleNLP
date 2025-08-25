import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import framework, core

from paddlenlp.ops import Topology
from paddlenlp.trainer import AutoTrainingArguments, PdArgumentParser
from paddlenlp.trainer.auto_trainer import AutoTrainer
from paddlenlp.trainer.trainer_utils import IntervalStrategy, _get_distributed_seeds
from paddlenlp.transformers import (
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    LlamaConfig,
    LlamaForCausalLM3DAuto,
    LlamaForCausalLMNet,
    LlamaPretrainingCriterion3DAuto,
    LlamaPretrainingCriterionNet,
)
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "llama": (LlamaConfig, LlamaForCausalLM3DAuto, LlamaPretrainingCriterion3DAuto),
    "llama_network": (LlamaConfig, LlamaForCausalLMNet, LlamaPretrainingCriterionNet),
}

from paddlenlp.trainer.utils.doc import add_start_docstrings
from paddlenlp.utils.tools import get_env_device

from paddle.io import Dataset, DistributedBatchSampler
import numpy as np

from paddlenlp.experimental.galvatron.profiler.runtime_profiler import RuntimeProfilerArguments

class DummyDataset(Dataset):
    def __init__(self, vocab_size, seq_length):
        super(DummyDataset, self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.generate_dummy_data()
    
    def generate_dummy_data(self):
        self.dataset_size = 512 # Temporarily set to 512
        self.input_list = []
        self.label_list = []
        
        for _ in range(self.dataset_size):
            single_sentence_length = np.random.randint(1, self.seq_length + 1) # [1, seq_length + 1)
            input = np.random.randint(0, self.vocab_size, size=(self.seq_length,), dtype=np.int64)
            input[single_sentence_length:] = 0
            label = np.zeros_like(input)
            label[:-1] = input[1:self.seq_length]
            self.input_list.append(input)
            self.label_list.append(label)
    
    def __getitem__(self, idx):
        idx = idx % self.dataset_size
        if idx >= self.dataset_size:
            raise IndexError("Index out of range")
        return {"input_ids": self.input_list[idx], "labels": self.label_list[idx]}
    
    def __len__(self):
        return self.dataset_size * 20 * 8 * 4096

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, seq_length=1024, split="train", cache_dir="./datacache"):
        super(WikiTextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.split = split
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.load_and_process_data()
    
    def get_cache_filename(self):
        """生成缓存文件名"""
        import hashlib
        # 使用tokenizer的vocab_size和seq_length作为缓存标识
        cache_key = f"{self.split}_{self.seq_length}_{getattr(self.tokenizer, 'vocab_size', 'unknown')}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        # return os.path.join(self.cache_dir, f"wikitext_{cache_hash}.npz")
        return os.path.join(self.cache_dir, 'wikitext.npz')
    
    def load_and_process_data(self):
        cache_file = self.get_cache_filename()
        
        # 尝试从缓存加载
        if os.path.exists(cache_file):
            try:
                logger.info(f"Loading cached dataset from {cache_file}")
                cached_data = np.load(cache_file)
                # assert len(cached_data['input_ids_0']) == 8192
                # small_seq_len = 8192
                small_seq_len = len(cached_data['input_ids_0'])
                seqs_per_big = self.seq_length // small_seq_len
                input_keys = sorted([k for k in cached_data.files if k.startswith('input_ids_')])
                label_keys = sorted([k for k in cached_data.files if k.startswith('labels_')])
                
                self.input_ids_list = []
                self.labels_list = []
                
                for i in range(0, len(input_keys), seqs_per_big):
                    if i + seqs_per_big >= len(input_keys):
                        logger.info(f"最后一批次 长度不匹配")
                        break
                    input_batch_keys = input_keys[i:i + seqs_per_big]
                    label_batch_keys = label_keys[i:i + seqs_per_big]

                    input_big = np.concatenate([cached_data[k] for k in input_batch_keys], axis=0)
                    label_big = np.concatenate([cached_data[k] for k in label_batch_keys], axis=0)

                    assert input_big.shape[0] == self.seq_length, f"拼接后 input 长度不对: {input_big.shape[0]}"
                    assert label_big.shape[0] == self.seq_length, f"拼接后 label 长度不对: {label_big.shape[0]}"

                    self.input_ids_list.append(input_big)
                    self.labels_list.append(label_big)
                    
                logger.info(f"成功拼接得到 {len(self.input_ids_list)} 个样本，每个长度 {self.seq_length}")

                # print(f'[linguangming] {len(cached_data.files)}')
                # self.input_ids_list = [cached_data[f'input_ids_{i}'] for i in range(len(cached_data.files)//2)]
                # self.labels_list = [cached_data[f'labels_{i}'] for i in range(len(cached_data.files)//2)]
                # print(f'input_id_len is  {len(self.input_ids_list)}')
                # print(f'input_ids_seq is {len(self.input_ids_list[0])}')
                # logger.info(f"Successfully loaded {len(self.input_ids_list)} cached samples")
                # exit(0)
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, will recreate dataset")
        # exit(0)
        # 如果没有缓存或缓存加载失败，重新处理数据
        if hf_load_dataset is None:
            raise ImportError("Please install datasets: pip install datasets")
            
        try:
            # Load WikiText-103-raw dataset
            logger.info(f"Loading WikiText-103-raw dataset, split: {self.split}")
            dataset = hf_load_dataset("wikitext", "wikitext-103-raw-v1", split=self.split)
            logger.info(f"Downloaded {len(dataset)} raw samples")
            
            # 为了加速，只使用前10%的数据进行测试
            if self.split == "train":
                dataset = dataset.select(range(min(len(dataset), len(dataset) // 10)))
                logger.info(f"Using subset of {len(dataset)} samples for faster processing")
            
            # Filter and concatenate all non-empty texts
            valid_texts = [example["text"] for example in dataset if example["text"] and example["text"].strip()]
            logger.info(f"Found {len(valid_texts)} valid text samples")
            
            # Concatenate all texts with proper separation
            all_text = "\n\n".join(valid_texts)
            logger.info(f"Total text length: {len(all_text)} characters")
            
            # Tokenize the concatenated text
            logger.info("Tokenizing text...")
            tokenized = self.tokenizer(
                all_text,
                return_attention_mask=False,
                return_token_type_ids=False,
                padding=False,
                truncation=False
            )["input_ids"]
            logger.info(f"Tokenized to {len(tokenized)} tokens")
            
            # Split into chunks of seq_length
            self.input_ids_list = []
            self.labels_list = []
            
            for i in range(0, len(tokenized) - self.seq_length, self.seq_length):
                input_ids = tokenized[i:i + self.seq_length]
                labels = tokenized[i + 1:i + self.seq_length + 1]
                
                # Ensure we have exactly seq_length tokens
                if len(input_ids) == self.seq_length and len(labels) == self.seq_length:
                    self.input_ids_list.append(np.array(input_ids, dtype=np.int64))
                    self.labels_list.append(np.array(labels, dtype=np.int64))
            
            logger.info(f"Created {len(self.input_ids_list)} training samples with seq_length={self.seq_length}")
            
            # 保存到缓存
            self._save_to_cache(cache_file)
            
        except Exception as e:
            logger.error(f"Error loading WikiText dataset: {e}")
            raise
    
    def _save_to_cache(self, cache_file):
        """保存处理后的数据到缓存"""
        try:
            logger.info(f"Saving dataset to cache: {cache_file}")
            cache_dict = {}
            for i, (input_ids, labels) in enumerate(zip(self.input_ids_list, self.labels_list)):
                cache_dict[f'input_ids_{i}'] = input_ids
                cache_dict[f'labels_{i}'] = labels
            
            np.savez_compressed(cache_file, **cache_dict)
            logger.info(f"Successfully saved {len(self.input_ids_list)} samples to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def __getitem__(self, idx):
        if idx >= len(self.input_ids_list):
            idx %= len(self.input_ids_list)
            # raise IndexError("Index out of range")
        return {
            "input_ids": self.input_ids_list[idx], 
            "labels": self.labels_list[idx]
        }
    
    def __len__(self):
        return len(self.input_ids_list) * 5120

def get_tokenizer():
    """获取并初始化tokenizer"""
    from transformers import AutoTokenizer
    import os
    
    # 使用已下载的本地tokenizer
    tokenizer_dir = "./tokenizers/llama3"
    
    if os.path.exists(tokenizer_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            logger.info(f"成功从本地路径加载 tokenizer: {tokenizer_dir}")
        except Exception as e:
            logger.error(f"从本地路径加载tokenizer失败: {e}")
            # 备用方案：尝试从远程加载
            logger.info("尝试从远程加载 NousResearch/Llama-2-7b-hf tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
            logger.info("成功从远程加载 tokenizer")
    else:
        # 直接从远程加载
        logger.info("本地tokenizer目录不存在，从远程加载 NousResearch/Llama-2-7b-hf tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
        logger.info("成功从远程加载 tokenizer")
   
    # 设置特殊tokens
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def create_real_dataset(tokenizer, seq_length=1024):
    """使用已有的tokenizer创建数据集"""
    # Create WikiText dataset
    train_dataset = WikiTextDataset(tokenizer, seq_length=seq_length, split="train")
    
    from paddlenlp.data import Stack
    def wikitext_collate_fn(data):
        stack_fn = Stack()
        input_ids = stack_fn([x["input_ids"] for x in data])
        labels = stack_fn([x["labels"] for x in data])
        return {"input_ids": input_ids, "labels": labels}
    
    return train_dataset, wikitext_collate_fn

@dataclass
@add_start_docstrings(AutoTrainingArguments.__doc__)
class PreTrainingArguments(AutoTrainingArguments):
    min_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Minimum learning rate deacyed to."},
    )
    decay_steps: float = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, will use the min_learning_rate."
        },
    )
    enable_linear_fused_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Enable fused linear grad add strategy, which will reduce elementwise add for grad accumulation in the backward of nn.Linear ."
        },
    )
    pipeline_schedule_mode: str = field(
        default="1F1B", metadata={"help": "The pipeline schedule mode, support FThenB, 1F1B, VPP and Eager-1F1B."}
    )
    sr: Optional[int] = field(default=0, metadata={"help": "The count of chunks without recompute."})
    virtual_pipeline_seg_method: str = field(
        default="LlamaDecoderLayerAuto", metadata={"help": "The seg method of spliting pp layer for virtual pipeline."}
    )
    # NOTE(gongenlei): new add autotuner_benchmark
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.enable_auto_parallel

        # NOTE(gongenlei): new add autotuner_benchmark
        if self.autotuner_benchmark:
            self.max_steps = 5
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.report_to = []
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO

        logger.info(self.strategy)

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluating.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    input_dir: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    split: str = field(default="949,50,1", metadata={"help": "Train/valid/test data split."})

    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    share_folder: bool = field(
        default=False,
        metadata={"help": "Use share folder for data dir and output dir on multi machine."},
    )

    data_impl: str = field(default="mmap", metadata={"help": "The format of the preprocessed data."})
    skip_warmup: bool = field(
        default=True,
        metadata={"help": "Whether to skip the warmup process of mmap files."},
    )
    data_cache: str = field(default=None, metadata={"help": "The path of the cached dataset."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to pre-train from.
    """

    model_type: Optional[str] = field(
        default="llama", metadata={"help": "Only support for llama pre-training for now."}
    )
    model_name_or_path: str = field(
        default="__internal_testing__/tiny-random-llama",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    use_fast_layer_norm: bool = field(
        default=False,
        metadata={"help": "GPT3 model, use fast layernorm"},
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    vocab_size: Optional[int] = field(
        default=None,
        metadata={
            "help": ".Vocabulary size of the Llama model. Defines the number of different tokens that can be represented by the `inputs_ids`"
        },
    )
    hidden_size: Optional[int] = field(default=None, metadata={"help": "Dimension of the hidden representations."})
    seq_length: Optional[int] = field(default=None, metadata={"help": "sequence length."})
    intermediate_size: Optional[int] = field(default=None, metadata={"help": "Dimension of the MLP representations."})
    num_hidden_layers: Optional[int] = field(
        default=None, metadata={"help": "Number of hidden layers in the Transformer encoder."}
    )
    num_attention_heads: Optional[int] = field(
        default=None,
        metadata={"help": "Number of attention heads for each attention layer in the Transformer encoder."},
    )
    num_key_value_heads: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of key and value heads for each attention layer in the Transformer encoder. "
            "If not set, it will be set to num_attention_heads // 2."
        },
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "use_flash_attention"},
    )
    use_fused_rms_norm: bool = field(
        default=False,
        metadata={"help": "llama, use_fused_rms_norm"},
    )
    fuse_attention_qkv: bool = field(
        default=False,
        metadata={"help": "whether to fuse attention qkv"},
    )
    fuse_attention_ffn: bool = field(
        default=False,
        metadata={"help": "whether to fuse first up and gate proj in mlp block"},
    )
    recompute_granularity: str = field(
        default="full",
        metadata={"help": "Choose among ['full', 'core_attn', 'full_attn']"},
    )
    virtual_pp_degree: int = field(
        default=1,
        metadata={"help": "virtual_pp_degree"},
    )
    continue_training: bool = field(
        default=False,
        metadata={
            "help": "Pre-training from existing paddlenlp model weights. Default False and model will train from scratch. If set True, the model_name_or_path argument must exist in the paddlenlp models."
        },
    )
    use_fused_rope: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable rope fusion or not."},
    )
    no_recompute_layers: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Specify the full transformer layers that should not be recomputed."},
    )
    pp_recompute_interval: int = field(
        default=1,
        metadata={
            "help": "The interval for the number of layers at which recomputation occurs. A value of 0 indicates no recomputation. Default is 0."
        },
    )
    recompute_use_reentrant: bool = field(
        default=False,
        metadata={"help": "recompute_use_reentrant"},
    )

def create_dataset(vocab_size=32000, seq_length=1024):
    train_dataset = DummyDataset(vocab_size, seq_length)
    
    from paddlenlp.data import Stack
    def dummy_collate_fn(data):
        stack_fn = Stack()
        input_ids = stack_fn([x["input_ids"] for x in data])
        labels = stack_fn([x["labels"] for x in data])
        return {"input_ids": input_ids, "labels": labels}
    
    return train_dataset, dummy_collate_fn

class PretrainingTrainer(AutoTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_pretraining = True

    def _wrap_for_dist_loader(self, train_dataloader):
        dist_loader = super()._wrap_for_dist_loader(train_dataloader)
        dist_loader._input_keys = ["input_ids", "labels"]
        return dist_loader

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None:
            return None

        total_batch_size_per_acc_step = self.args.per_device_train_batch_size * self.args.dataset_world_size  # per_device_train_batch_size是mbsz dataset_world_size是dp*sdp
        total_batch_size = total_batch_size_per_acc_step

        # In llm/llama/run_pretrain.py, it uses paddlenlp.utils.batch_sampler.DistributedBatchSampler,
        # which does no shuffle when shuffle is set True.
        sampler = paddle.io.BatchSampler(
            dataset=self.train_dataset,
            shuffle=False,
            batch_size=total_batch_size,
            drop_last=self.args.dataloader_drop_last,
        )
        sampler._acc_steps = self.args.gradient_accumulation_steps
        return sampler

def init_seed(seed: int = 1234, args=None):
    if args is None:
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
    else:
        assert not args.use_hybrid_parallel and args.enable_auto_parallel
        if dist.get_world_size() > 1:
            if args.hybrid_parallel_topo_order is None or args.hybrid_parallel_topo_order == "pp_first":
                order = ["pp", "dp", "sharding", "mp", "sep"]
            elif args.hybrid_parallel_topo_order == "sharding_first":
                order = ["dp", "sharding", "pp", "mp", "sep"]
            topo = Topology(
                dist.get_rank(),
                dist.get_world_size(),
                dp_degree=args.dataset_world_size,
                pp_degree=args.pipeline_parallel_degree,
                mp_degree=args.tensor_parallel_degree,
                sharding_degree=1,  # auto_parallel's sharding is not orthogonal with dp, mp and pp
                order=order,
            )

            global_seed, local_seed, random_seed = _get_distributed_seeds(args.seed, topo)

            paddle.seed(local_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

            logger.info(
                "The global seed is set to {}, local seed is set to {} and "
                "random seed is set to {}.".format(global_seed, local_seed, random_seed)
            )
        else:
            random.seed(args.seed)
            np.random.seed(args.seed)
            paddle.seed(args.seed)

def runtime_profiler_initalize_manully(runtime_profiler_args:RuntimeProfilerArguments, training_args, model_args):
    runtime_profiler_args.global_rank = dist.get_rank()
    runtime_profiler_args.pp_degree = training_args.pipeline_parallel_degree
    runtime_profiler_args.tp_degree = training_args.tensor_parallel_degree
    runtime_profiler_args.dp_degree = training_args.dataset_world_size
    runtime_profiler_args.runtime_profiler_to_static = training_args.to_static
    runtime_profiler_args.runtime_profiler_recompute = training_args.recompute
    
    runtime_profiler_args.dp_rank = training_args.dataset_rank
    runtime_profiler_args.tp_rank = training_args.tensor_parallel_rank
    runtime_profiler_args.pp_rank = training_args.pipeline_parallel_rank

    runtime_profiler_args.model_name = "llama"
    runtime_profiler_args.layernum = model_args.num_hidden_layers
    runtime_profiler_args.seq_len = model_args.seq_length
    runtime_profiler_args.global_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.dataset_world_size
    runtime_profiler_args.mixed_precision = 'bf16' if training_args.bf16 else 'fp16' if training_args.fp16 else 'fp32'

def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, PreTrainingArguments, RuntimeProfilerArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, runtime_profiler_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, runtime_profiler_args = parser.parse_args_into_dataclasses()

    if data_args.data_cache is not None:
        os.makedirs(data_args.data_cache, exist_ok=True)

    init_seed(args=training_args)
    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # Log model and data config
    runtime_profiler_initalize_manully(runtime_profiler_args, training_args, model_args)
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(training_args, "Training")
    training_args.print_config(runtime_profiler_args, "RuntimeProfile")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    config_class, model_class, criterion_class = MODEL_CLASSES[model_args.model_type]

    config = config_class()
    config.num_hidden_layers = model_args.num_hidden_layers
    config.intermediate_size = model_args.intermediate_size
    config.vocab_size = model_args.vocab_size
    config.hidden_size = model_args.hidden_size
    config.seq_length = model_args.seq_length
    config.max_position_embeddings = config.seq_length # use seq_length as max_position_embeddings
    config.num_attention_heads = model_args.num_attention_heads
    config.num_key_value_heads = model_args.num_key_value_heads if model_args.num_key_value_heads is not None else config.num_key_value_heads
    config.use_fast_layer_norm = model_args.use_fast_layer_norm
    if model_args.no_recompute_layers is not None:
        model_args.no_recompute_layers.sort()
    config.use_flash_attention = model_args.use_flash_attention
    config.use_fused_rms_norm = model_args.use_fused_rms_norm
    config.fuse_attention_qkv = model_args.fuse_attention_qkv
    config.fuse_attention_ffn = model_args.fuse_attention_ffn
    config.recompute_granularity = model_args.recompute_granularity
    config.virtual_pp_degree = model_args.virtual_pp_degree
    config.sequence_parallel = training_args.sequence_parallel
    config.fuse_sequence_parallel_allreduce = training_args.fuse_sequence_parallel_allreduce
    config.use_fused_rope = model_args.use_fused_rope
    config.no_recompute_layers = model_args.no_recompute_layers
    config.pp_recompute_interval = model_args.pp_recompute_interval
    config.recompute_use_reentrant = model_args.recompute_use_reentrant
    config.use_recompute = training_args.recompute
    config.tensor_parallel_degree = training_args.tensor_parallel_degree
    config.tensor_parallel_rank = training_args.tensor_parallel_rank
    config.sharding_parallel_degree = training_args.sharding_parallel_degree
    if training_args.strategy.pipeline.enable and config.virtual_pp_degree > 1:
        pipeline = training_args.strategy.pipeline
        pipeline.vpp_degree = config.virtual_pp_degree
        pipeline.vpp_seg_method = training_args.virtual_pipeline_seg_method

    print("[auto-parallel] Model Config:", config)

    with paddle.LazyGuard():
        model = model_class.from_config(config, dtype="float32")
        criterion = criterion_class(config)

    print("[auto-parallel] Model initialized")
    print(f'model is {model}')

    if training_args.recompute: # As described in the corresponding model definition, Recompute defaults to False and is controlled by Trainer
        def fn(layer):
            if hasattr(layer, "enable_recompute") and (layer.enable_recompute is False or layer.enable_recompute == 0):
                layer.enable_recompute = True
        model.apply(fn)

    # Create the learning_rate scheduler and optimizer
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    if training_args.warmup_steps > 0:
        warmup_steps = training_args.warmup_steps
    else:
        warmup_steps = training_args.warmup_ratio * training_args.max_steps

    lr_scheduler = None
    if training_args.lr_scheduler_type.value == "cosine":
        lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )
    elif training_args.lr_scheduler_type.value == "linear":
        lr_scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )
    
    tokenizer = get_tokenizer()
    train_dataset, data_collator = create_real_dataset(tokenizer, model_args.seq_length)
    # train_dataset, data_collator = create_dataset(config.vocab_size, config.seq_length)
    
    # [workflow] 以下为模型封装代码
    trainer = PretrainingTrainer(
        model=model,
        criterion=criterion,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        optimizers=(None, lr_scheduler),
        runtime_profiler_args=runtime_profiler_args,
    )
    print("[auto-parallel] PretrainingTrainer OK")    
    
    print('After model initialization, current allocated memory')
    current_device = framework._current_expected_place_()
    max_memory_allocated = core.device_memory_stat_peak_value("Allocated", current_device.get_device_id()) / 2**20
    current_memory_allocated = core.device_memory_stat_current_value("Allocated", current_device.get_device_id()) / 2**20
    print(f"Max memory allocated: {max_memory_allocated} MB")
    print(f"Current memory allocated: {current_memory_allocated} MB")
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        
if __name__ == "__main__":
    main()