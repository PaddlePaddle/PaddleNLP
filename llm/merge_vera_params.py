import argparse
import copy
import os

import paddle
# paddle.enable_static()
from paddlenlp.peft import VeRAConfig, VeRAModel

try:
    from paddle.nn.quant import weight_dequantize, weight_quantize
except:
    weight_dequantize = None
    weight_quantize = None
try:
    from paddlenlp.quantization.qlora import qlora_weight_quantize_dequantize
except:
    qlora_weight_quantize_dequantize = None

from paddlenlp.quantization.quantization_config import QuantizationConfig
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.transformers.utils import device_guard
from paddlenlp.utils.env import CONFIG_NAME


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, help="The directory of pretrained model.")
    parser.add_argument(
        "--vera_path", default='',  help="The directory of VeRA parameters. Default to None"
    )
    parser.add_argument(
        "--merge_vera_model_path",
        default='',
        help="The directory of merged parameters. Default to None",
    )
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument(
        "--low_gpu_mem", type=bool, default=True, help="Whether to use low gpu memory. Default to False"
    )
    return parser.parse_args()


def weight_process(name, quant_config, vera_config, state_dict):
    weight = state_dict.pop(name + ".weight").cuda()
    if quant_config.weight_quantize_algo is None:
        pass
    elif quant_config.weight_quantize_algo in ["nf4", "fp4"]:
        weight = qlora_weight_quantize_dequantize(
            weight,
            quant_algo=quant_config.weight_quantize_algo,
            double_quant=quant_config.weight_double_quant,
            block_size=quant_config.weight_blocksize,
            double_quant_block_size=quant_config.weight_double_quant_block_size,
        )
    elif quant_config.weight_quantize_algo in ["weight_only_int8"]:
        out, scale = weight_quantize(weight, algo=quant_config.weight_quantize_algo)
        weight = weight_dequantize(out, scale)
    else:
        raise ValueError(f"quant_config.weight_quantize_algo {quant_config.weight_quantize_algo} is not supported.")
    lora_A = state_dict.pop(name + ".lora_A").cuda()
    lora_B = state_dict.pop(name + ".lora_B").cuda()
    vera_b = state_dict.pop(name + ".vera_b").cuda()
    vera_d = state_dict.pop(name + ".vera_d").cuda()
    diag_b = paddle.diag(vera_b)
    diag_d = paddle.diag(vera_d)
    
    
    print(name)
    if name == 'llama.layers.0.self_attn.q_proj':
        print('weight', weight)
        print('lora-A', lora_A)
        print('lora_B', lora_B )
        print('vera_b', vera_b)
        print('vera_d', vera_d)
        scaling = vera_config.lora_alpha / vera_config.r
        # print('scaling:', scaling)
        state_dict[name + ".weight"] = (weight + lora_A @ diag_d @ lora_B @ diag_b * scaling).cpu()
        # print('merged weight', state_dict[name + ".weight"])
        # exit(0)
        
    scaling = vera_config.lora_alpha / vera_config.r
    # print('scaling:', scaling)
    state_dict[name + ".weight"] = (weight + lora_A @ diag_d @ lora_B @ diag_b * scaling).cpu()


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)

    vera_config = VeRAConfig.from_pretrained(args.vera_path)
    if vera_config.base_model_name_or_path is None:
        if args.model_name_or_path is not None:
            raise ValueError("We can not find a valid model_name_or_path.")
        else:
            vera_config.base_model_name_or_path = args.model_name_or_path

    if os.path.isfile(os.path.join(args.vera_path, CONFIG_NAME)):
        config = AutoConfig.from_pretrained(args.vera_path)
    elif args.model_name_or_path is not None:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            f"We can not find config.json in vera_path: {args.vera_path} or find a valid model_name_or_path."
        )
    config.dtype = vera_config.dtype
    if (
        vera_config.dtype == "bfloat16" or config.quantization_config.weight_quantize_algo in ["nf4", "fp4"]
    ) and args.device == "cpu":
        raise ValueError("We can not apply bfloat16 or nf4/fp4 lora merge on cpu.")

    if args.low_gpu_mem and args.device == "gpu":
        quant_config = copy.deepcopy(config.quantization_config)
        config.quantization_config = QuantizationConfig()
        vera_config.merge_weights = False
        # with device_guard(): 会导致svd无法进行分解
        print('loading base model from lora config', vera_config.base_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            vera_config.base_model_name_or_path,
            config=config,
            low_cpu_mem_usage=True,
        )
        model = VeRAModel.from_pretrained(model=model, vera_path=args.vera_path, vera_config=vera_config)
       
        print('after from pretrained model:', model)
        # params = dict(model.named_parameters())
        # print(params.keys())
        # print('llama.layers.0.self_attn.q_proj.lora_B', params['model.llama.layers.31.self_attn.v_proj.lora_B'])
        
        model.eval()
        model_state_dict = model.model.state_dict()
        lora_name_list = []
        for key in model_state_dict.keys():
            if "lora_A" in key:
                lora_name_list.append(key[:-7])
        print('lora_name_list', lora_name_list)
        for name in lora_name_list:
            weight_process(name, quant_config, vera_config, model_state_dict)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            vera_config.base_model_name_or_path,
            config=config,
            low_cpu_mem_usage=True,
        )
        vera_config.merge_weights = True
        model = VeRAModel.from_pretrained(model=model, vera_path=args.vera_path, vera_config=vera_config)
        
        model_state_dict = model.model.state_dict()
        for key in list(model_state_dict):
            if "lora" in key:
                del model_state_dict[key]
            if "quant" in key:
                del model_state_dict[key]
        model.model.config.quantization_config = QuantizationConfig()
    model.model.save_pretrained(args.merge_vera_model_path, state_dict=model_state_dict)

    tokenizer = AutoTokenizer.from_pretrained(vera_config.base_model_name_or_path)
    tokenizer.save_pretrained(args.merge_vera_model_path)


if __name__ == "__main__":
    merge()
