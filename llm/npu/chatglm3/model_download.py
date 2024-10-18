# encoding=utf-8
import os
import json
from tqdm import tqdm
import paddle
import argparse
from paddlenlp.utils.download import resolve_file_path
from paddlenlp.transformers import AutoModelForCausalLM

MODEL_FILE_NAMES = ["config.json", "tokenizer.model", "tokenizer_config.json", "chat_template.json",
                    "model_state.pdparams","special_tokens_map.json"]


def model_download(model_name, output_path):
    """
    参数:
    model_name (str): 模型名称。
    file_path (str): 文件保存路径，指定模型文件的下载位置。

    返回:
    无
    """
    # 定义缓存目录
    cache_dir = os.path.join(output_path, model_name)

    for filename in MODEL_FILE_NAMES:
        temp_path = os.path.join(cache_dir, filename)
        target_path = os.path.join(output_path, filename)

        if os.path.exists(target_path):
            continue

        resolve_file_path(model_name, filenames=filename, cache_dir=output_path)

        if os.path.exists(temp_path):
            os.rename(temp_path, target_path)
        else:
            print("Not Found {} file!".format(filename))

    # 移除缓存目录（如果为空）
    if not os.listdir(cache_dir):
        os.rmdir(cache_dir)

    # 修改模型配置文件
    config_dict = {"model_type": "chatglm3", "architectures": ["ChatGLMv3ForCausalLM"]}
    update_json_config(os.path.join(args.model_save_path, "config.json"), config_dict)

    # 修改tokenizer配置文件
    tokenizer_dict = {"tokenizer_class": "ChatGLMv3Tokenizer"}
    update_json_config(os.path.join(args.model_save_path, "tokenizer_config.json"), tokenizer_dict)

def update_json_config(config_path, updates):
    """
    修改模型配置文件的部分参数并保存。

    参数:
    - config_path (str): 配置文件的路径。
    - updates (dict): 要更新的参数字典，键为参数名，值为新的参数值。

    返回:
    - None
    """
    # 读取现有配置文件
    with open(config_path, 'r') as file:
        config = json.load(file)

    # 更新配置参数
    for key, value in updates.items():
        config[key] = value

    # 保存更新后的配置文件
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

def convert_pdparams(param_path):
    files = os.listdir(param_path)
    if "model_state.pdparams" in files:
        params = paddle.load(os.path.join(param_path, "model_state.pdparams"))
    else:
        raise ValueError("The model_state.pdparams file is not found!")

    model_state = {}
    n_heads = 32
    head_dim = 128
    num_key_value_heads = 2

    for k, v in tqdm(params.items()):
        if "chatglm_v2.encoder" in k:
            k = k.replace("chatglm_v2.encoder", "transformer")

            if "query_key_value.weight" in k:
                q_proj = k.replace("query_key_value", "q_proj")
                k__proj = k.replace("query_key_value", "k_proj")
                v_proj = k.replace("query_key_value", "v_proj")

                # 定义分割的比例
                num_or_sections = [n_heads * head_dim, num_key_value_heads * head_dim, num_key_value_heads * head_dim]

                # 使用 paddle.split 进行分割
                model_state[q_proj], model_state[k__proj], model_state[v_proj] = paddle.split(
                    v,
                    num_or_sections=num_or_sections,
                    axis=-1,
                )

            elif "query_key_value.bias" in k:
                q_proj = k.replace("query_key_value", "q_proj")
                k__proj = k.replace("query_key_value", "k_proj")
                v_proj = k.replace("query_key_value", "v_proj")
                model_state[q_proj] = v[:n_heads * head_dim]
                model_state[k__proj] = v[n_heads * head_dim:n_heads * head_dim + num_key_value_heads * head_dim]
                model_state[v_proj] = v[n_heads * head_dim + num_key_value_heads * head_dim:]

            else:
                model_state[k.replace("chatglm_v2.encoder", "transformer")] = v

        elif "output_layer" in k:
            model_state["output_layer.weight"] = v

        elif "chatglm_v2.embedding" in k:

            model_state[k.replace("chatglm_v2.embedding", "transformer")] = v

        elif "chatglm_v2" in k:
            model_state[k.replace("chatglm_v2", "transformer")] = v

        else:
            model_state[k] = v
    os.rename(os.path.join(param_path, "model_state.pdparams"), os.path.join(param_path, "model_state_old.pdparams"))
    paddle.save(model_state, os.path.join(param_path, "model_state.pdparams"))
    print("Convert model successfully!")

def main(args):

    # 下载模型文件
    model_download(args.model_name, args.model_save_path)

    # 转换模型
    convert_pdparams(args.model_save_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_save_path, dtype="float16")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="THUDM/chatglm3-6b", type=str, required=False)
    parser.add_argument("--model_save_path", default="./model", type=str, required=False)
    args = parser.parse_args()
    main(args)