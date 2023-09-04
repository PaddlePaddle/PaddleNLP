import os
import requests
from PIL import Image

import paddle
from paddlenlp.transformers import MiniGPT4ForConditionalGeneration

# load MiniGPT4 moel and processor
# minigpt4_13b_path = "/root/.paddlenlp/models/Salesforce/minigpt4-vicuna-13b"
minigpt4_13b_path = "/root/paddlejob/workspace/env_run/zhengshifeng/vitllm/vit_model"
model = MiniGPT4ForConditionalGeneration.from_pretrained(minigpt4_13b_path, vit_dtype="float16")
model.eval()
#breakpoint()

# convert to static graph with specific input description
model = paddle.jit.to_static(
    model.generate_gpt,
    input_spec=[
    paddle.static.InputSpec(shape=[1, 3, 224, 224], dtype="float32"),  # image_features
    paddle.static.InputSpec(shape=[1, 1], dtype="int64"),  # first_input_ids
    paddle.static.InputSpec(shape=[1, 1], dtype="int64"),  # first_attention_mask
    # paddle.static.InputSpec(shape=[1], dtype="int64"),  # max_length
    ],
    )

# save to static model
save_path = "./checkpoints/infer"
paddle.jit.save(model, save_path)
print(f"static model has been to {save_path}")