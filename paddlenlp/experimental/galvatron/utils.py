import json
import os
from typing import List
import sys
from dataclasses import dataclass, field, fields

def read_json_config(path):
    if os.path.exists(path) == False:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fp:
            fp.write("{}")
    return json.load(open(path, 'r', encoding="utf-8"))

def write_json_config(path, config):
    with open(path, 'w') as fp:
        json.dump(config, fp, indent=4)

def num2str(num, type:str):  # NOTE this function maybe have some bugs
    if type == 'seq':
        if isinstance(num, List) and len(num) == 1:
            num = num[0]        
        if isinstance(num, List):
            info = f'seq[{",".join(map(str, num))}]'
        else:
            info = f'seq{num}'
    elif type == 'layernum':
        info = 'layernum' + '[' + str(num) + ']'
    return info

@dataclass
class Strategy:
    pp_size: int = field(default=1, metadata={"help": "The number of processes to use for parallel processing."})
    tp_size: int = field(default=1, metadata={"help": "The number of threads to use for parallel processing."})
    dp_size: int = field(default=1, metadata={"help": "The number of data parallelism to use."})
    sharding_stage: int = field(default=0, metadata={"help": "The stage of sharding. 0: no sharding, 1: sharding1, 2: sharding2, 3: sharding3"})
    recompute: int = field(default=0, metadata={"help": "Whether to use recompute."})
    
    def serialize(self):
        text = f'pp{self.pp_size}_tp{self.tp_size}_dp{self.dp_size}_stage{self.sharding_stage}_recompute{self.recompute}'
        return text
    
    def deserialize(self, text):
        if isinstance(text, str):
            items = text.split('_')
            for item in items:
                if 'pp' in item:
                    self.pp_size = int(item.split('pp')[1])
                elif 'tp' in item:
                    self.tp_size = int(item.split('tp')[1])
                elif 'dp' in item:
                    self.dp_size = int(item.split('dp')[1])
                elif 'stage' in item:
                    self.sharding_stage = int(item.split('stage')[1])
                elif 'recompute' in item:
                    self.recompute = int(item.split('recompute')[1])
        elif isinstance(text, dict):
            self.pp_size = text.get('pp_size', self.pp_size)
            self.tp_size = text.get('tp_size', self.tp_size)
            self.dp_size = text.get('dp_size', self.dp_size)
            self.sharding_stage = text.get('sharding_stage', self.sharding_stage)
            self.recompute = text.get('recompute', self.recompute)
        elif isinstance(text, List):
            self.pp_size = text[0]
            self.tp_size = text[1]
            self.dp_size = text[2]
            self.sharding_stage = text[3]
            self.recompute = text[4]
        else:
            raise ValueError("Unsupported type for deserialization. Supported types are str, dict, and list.")
        
    def __str__(self):
        return self.serialize()

@dataclass
class LayerWiseStrategy:
    # layer info
    # layer_idx: int = field(default=-1, metadata={"help": "Layer index"})
    
    #  pipeline parallelism
    pp_size: int = field(default=-1, metadata={"help": "The number of processes to use for parallel processing."})
    # stage_idx: int = field(default=-1, metadata={"help": "The stage idx or pipeline parallel"})
    
    # tensor parallelism(with megatron-sp) or ulysses-sp
    tp_size: int = field(default=-1, metadata={"help": "The number of threads to use for parallel processing."})
    use_ulysses: int = field(default=-1, metadata={"help": "Whether to use Ulysses for layer-wise strategy."})
    
    # data parallelism
    dp_size: int = field(default=-1, metadata={"help": "The number of data parallelism to use."})
    sharding_stage: int = field(default=-1, metadata={"help": "The stage of sharding. 0: no sharding, 1: sharding1, 2: sharding2, 3: sharding3"})
    
    # recompute
    recompute: int = field(default=-1, metadata={"help": "Whether to use recompute."})
    
    def serialize(self) -> str:
        field_defs = fields(self)
        parts = []
        for field_def in field_defs:
            value = getattr(self, field_def.name)
            parts.append(f'{field_def.name}={value:<3}')
        return ",".join(parts)
    
    @classmethod
    def deserialize(cls, info):
        if isinstance(info, str):
            kwargs = {}
            for pair in info.split(","):
                pair = pair.strip()
                if pair == "":
                    assert False
                key, value = pair.split("=")
                if value.isdigit():
                    value = int(value)
                kwargs[key] = value
            return cls(**kwargs)
        elif isinstance(info, dict):
            return cls(**info)
        else:
            raise ValueError("Unsupported type for deserialization. Supported types are str, dict")
    
    def __str__(self):
        return self.serialize()

@dataclass
class OptimalSolution:
    pp_deg: int = field(default=1, metadata={"help": "The degree of pipeline parallelism."})
    layer_wise_strategies: List[LayerWiseStrategy] = field(default_factory=list, metadata={"help": "List of layer-wise strategies."})
    
    vocab_dp: int = field(default=1, metadata={"help": "The degree of vocab data parallelism."})
    vocab_sharding_stage: int = field(default=0, metadata={"help": "The stage of vocab sharding."})
    vocab_tp: int = field(default=1, metadata={"help": "The degree of vocab tensor parallelism."})
    vocab_sp: int = field(default=1, metadata={"help": "The degree of vocab sharding parallelism."})

    global_batch_size: int = field(default=1, metadata={"help": "The global batch size"})
    accumulate_steps: int = field(default=1, metadata={"help": "The accumulate steps"})
    
    def store(self, config_path):
        kwargs = {
            'pp_deg': self.pp_deg,
            'vocab_dp': self.vocab_dp,
            'vocab_sharding_stage': self.vocab_sharding_stage,
            'vocab_tp':  self.vocab_tp,
            'vocab_sp': self.vocab_sp,
            'global_batch_size': self.global_batch_size,
            'accumulate_steps': self.accumulate_steps,
            'dp_size_enc': ",".join(str(strategy.dp_size) for strategy in self.layer_wise_strategies),
            'sharding_stage_enc': ",".join(str(strategy.sharding_stage) for strategy in self.layer_wise_strategies),
            'tp_size_enc': ','.join(str(strategy.tp_size) for strategy in self.layer_wise_strategies),
            'use_ulysses_enc':  ','.join(str(strategy.use_ulysses) for strategy in self.layer_wise_strategies),
            'stage_idx_enc': ','.join(str(strategy.stage_idx) for strategy in self.layer_wise_strategies),
            'recompute_enc': ','.join(str(strategy.recompute) for strategy in self.layer_wise_strategies),
        }
        with open(config_path, 'w') as f:
            json.dump(kwargs, f, indent=4) 
        print(f'Optimal solution has stored to {config_path}!')

def get_current_all_args():
    args_dict = {}
    i = 0
    argv = sys.argv
    while i < len(argv):
        arg = argv[i]
        if arg.startswith('-'):
            if i + 1 < len(argv) and not argv[i + 1].startswith('-'):
                args_dict[arg] = argv[i + 1]
                i += 2  
            else:
                args_dict[arg] = True
                i += 1
        else:
            i += 1
    return args_dict