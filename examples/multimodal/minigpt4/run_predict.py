# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import numpy as np
import paddle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"
import requests
from PIL import Image
from paddlenlp.transformers import GPTTokenizer
from paddlenlp.transformers import LlamaTokenizer

from paddlenlp.transformers import MiniGPT4ForConditionalGeneration, MiniGPT4Processor


def predict(args):
    # load MiniGPT4 moel and processor
    model = MiniGPT4ForConditionalGeneration.from_pretrained(args.pretrained_name_or_path)
    model.eval()
    # processor = MiniGPT4Processor.from_pretrained(args.pretrained_name_or_path)
    print("load processor and model done!")

    # prepare model inputs for MiniGPT4
    # url = "https://paddlenlp.bj.bcebos.com/data/images/mugs.png"
    # image = Image.open(requests.get(url, stream=True).raw)

    # text = "describe this image"
    # prompt = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> <TextHere>###Assistant:"
    # inputs = processor([image], text, prompt)


    import numpy as np
    path_data = "./vit_numpy.npy"
    vit_data = np.load(path_data, allow_pickle=True)
    
    # vit_inputs = vit_data["input"]
    # vit_outputs = vit_data["output"]
    for i in range(200):
        vit_image = vit_data[i]
        inputs={}
        inputs["pixel_values"] = paddle.to_tensor(vit_image)
        inputs["first_input_ids"] = paddle.to_tensor([[1]])
        inputs["first_attention_mask"] = paddle.to_tensor([[1]])
        # inputs["max_len"] = paddle.to_tensor([20])
        
        # print("inputs", inputs)

        # import pdb;pdb.set_trace()

        # generate with MiniGPT4
        # breakpoint
        generate_kwargs = {
            "max_length": 300,
            "num_beams": 5,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "length_penalty": 0,
            "temperature": 1,
            "decode_strategy": "beam_search",
            "eos_token_id": [[2]],
        }
        outputs = model.generate_gpt(**inputs, **generate_kwargs)

        # print("outputs", outputs)
        tokenizer = LlamaTokenizer.from_pretrained("/root/paddlejob/workspace/env_run/zhengshifeng/vitllm/pr/PaddleNLP/examples/multimodal/minigpt4/vit_model")


        output = []
        for i in range(222):
            if outputs[0][0].numpy()[i] != 2 or outputs[0][0].numpy()[i] != 73:
                output.append(outputs[0][0].numpy()[i])
            if outputs[0][0].numpy()[i] == 2 or outputs[0][0].numpy()[i] == 73:
                break
        # print(output)
        msg = tokenizer.convert_tokens_to_string(np.array(output).tolist())
        print("Inference result: ", msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_name_or_path",
        default="your directory of minigpt4",
        type=str,
        help="The dir name of minigpt4 checkpoint.",
    )
    args = parser.parse_args()

    predict(args)
