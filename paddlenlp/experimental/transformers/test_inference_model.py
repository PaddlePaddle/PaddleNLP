


# 后面会删掉，仅提供示例
from inference_utils import PredictorArgument, ModelArgument
from paddlenlp.experimental.transformers.inference_model import InferenceModel

predictor_args = PredictorArgument()
model_args = ModelArgument()

predictor_args.model_name_or_path = "/root/paddlejob/workspace/env_run/output/gaoziyuan/paddlenllp_model/models/Qwen/Qwen2-7B"
predictor_args.max_length = 2048
predictor_args.dtype = "bfloat16"
predictor_args.total_max_length = 4096
predictor_args.inference_model = True
predictor_args.mode = "dynamic"
predictor_args.block_attn = True

nranks = 1
rank = 0

inference_model = InferenceModel(predictor_args, model_args, nranks, rank, load_model_from_ipc=True)
model = inference_model.get_model()

# 获取inference model 的 key\shape\type
inference_model.get_model_static_info()

# 获取qwen2 训练和推理的key映射关系
train_to_infer, infer_to_train = inference_model.get_qwen2_train_infer_keys_map()

keys_all = model.state_dict().keys()
for k, v in infer_to_train.items():
    if k not in keys_all:
        print("missing key is :", k)
        print("missing v is :", v)