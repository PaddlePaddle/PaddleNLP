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
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"

import paddle
from paddle.distributed import fleet

from paddlenlp.transformers import (
    ChatGLMv2Config,
    ChatGLMv2Tokenizer,
)

from paddlenlp.transformers import ChatGLMv2ForCausalLM

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="THUDM/chatglm2-6b", help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=1280, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=1280, help="The batch size of data.")
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class Predictor(object):
    def __init__(self, args=None, tokenizer=None, model=None, **kwargs):
        if args is None:
            self.tokenizer = tokenizer
            self.model = model
            self.src_length = kwargs["src_length"]
            self.tgt_length = kwargs["tgt_length"]
        else:
            self.tokenizer = ChatGLMv2Tokenizer.from_pretrained(args.model_name_or_path)
            self.batch_size = args.batch_size
            self.args = args
            self.src_length = self.args.src_length
            self.tgt_length = self.args.tgt_length

            tensor_parallel_degree = paddle.distributed.get_world_size()
            tensor_parallel_rank = 0
            if tensor_parallel_degree > 1:
                strategy = fleet.DistributedStrategy()
                strategy.hybrid_configs = {
                    "dp_degree": 1,
                    "mp_degree": tensor_parallel_degree,
                    "pp_degree": 1,
                    "sharding_degree": 1,
                }
                fleet.init(is_collective=True, strategy=strategy)
                hcg = fleet.get_hybrid_communicate_group()
                tensor_parallel_rank = hcg.get_model_parallel_rank()

            config = ChatGLMv2Config.from_pretrained(args.model_name_or_path)
            dtype = config.dtype if config.dtype is not None else config.paddle_dtype

            self.model = ChatGLMv2ForCausalLM.from_pretrained(
                args.model_name_or_path,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                dtype=dtype,
            )
        self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="pd",
            padding=True,
            max_length=self.src_length,
            truncation=True,
            truncation_side="left",
        )
        return inputs

    def infer(self, inputs):

        # static_model = paddle.jit.to_static(
        #     self.model.encoder.layers[0],
        #     input_spec=[
        #         paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        #         paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # attention_mask
        #         paddle.static.InputSpec(shape=[None, None], dtype="int64"),  #position_ids
        #     ],
            
        # )
        # paddle.jit.save(static_model, "./static_model/inference")

        # static_model = paddle.jit.to_static(
        #     self.model.haha,
        #     input_spec=[
        #         paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        #         paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # attention_mask
        #         paddle.static.InputSpec(shape=[None, None], dtype="int64"),  #position_ids
        #     ],
            
        # )
        # paddle.jit.save(static_model, "./static_model/inference")
        # #exit(0)
        # print(self.tgt_length)
        # print(self.tokenizer.bos_token_id)
        # print(self.tokenizer.eos_token_id)
        # print(self.tokenizer.pad_token_id)
        result = self.model.generate(
            **inputs,
            decode_strategy="sampling",
            top_k=1,
            max_length=self.tgt_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
        )
        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            res = res.strip("\n")
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        #print(input_map)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predictor(args)
    all_texts = [
        #"[Round 0]\n问：你好\n答：你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。\n[Round 1]\n问：晚上睡不着应该怎么办\n答：",
    "<0>山东师范大学博士学位论文##129,16,199,21</0><1>构“大变”的成语。异体成语的运用体现出这一时期很多成语的构成要素并不十分固定，##36,30,287,35</1><2>成语结构也并不十分稳固，呈现出成语在结构上不断发生演变的过渡性特征，以及近代汉##36,39,291,44</2><3>语词汇向现代汉语词汇发展过程中的过渡性特征。##36,49,177,54</3><4>本章小结##146,62,180,69</4><5>本章我们在借鉴前人对成语的研究成果与界定的基础上，提出了对成语的界定：成语##48,77,291,83</5><6>是人们长期以来相沿习用的，意义相对完整、结构相对稳定的固定词组。成语的典型表现##36,87,291,92</6><7>形式为四字结构，多具有书面语色彩。进而提出了异体成语的定义：异体成语是整体意义##36,96,290,102</7><8>基本相同，形式上至少有一个相同的构成要素并且其他相异要素存在着互相替换的意义关##36,106,291,111</8><9>系，具有相同语法性质，结构稳定的一组词汇类聚。清末民初时期出现在白话报刊中的异##36,115,291,121</9><10>体成语，便是清末民初异体成语。清末民初异体成语中在当时使用频率最高的变体我们称##36,125,291,130</10><11>为清末民初异体成语的通体，其他称为清末民初异体成语的变体。##36,135,223,140</11><12>清末民初异体成语可以从形式、语义、结构等方面分类分析，总结分布特征。清末民##48,144,291,149</12><13>初异体成语在数量与分布上具有繁复性，在类型上具有多样化特征，在风格色彩上具有白##35,153,290,159</13><14>话化特征，在整体运用上具有过渡性特征等。##36,163,164,168</14><15>43##36,296,44,300</15><16>万方数据##26,310,48,315</16>",
        "[Round 0]\n问：你好\n答：你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。\n[Round 1]\n问：晚上睡不着应该怎么办\n答：",
     #"<0>山东师范大学博士学位论文##129,16,199,21</0><1>构“大变”的成语。异体成语的运用体现出这一时期很多成语的构成要素并不十分固定，##36,30,287,35</1><2>成语结构也并不十分稳固，呈现出成语在结构上不断发生演变的过渡性特征，以及近代汉##36,39,291,44</2><3>语词汇向现代汉语词汇发展过程中的过渡性特征。##36,49,177,54</3><4>本章小结##146,62,180,69</4><5>本章我们在借鉴前人对成语的研究成果与界定的基础上，提出了对成语的界定：成语##48,77,291,83</5><6>是人们长期以来相沿习用的，意义相对完整、结构相对稳定的固定词组。成语的典型表现##36,87,291,92</6><7>形式为四字结构，多具有书面语色彩。进而提出了异体成语的定义：异体成语是整体意义##36,96,290,102</7><8>基本相同，形式上至少有一个相同的构成要素并且其他相异要素存在着互相替换的意义关##36,106,291,111</8><9>系，具有相同语法性质，结构稳定的一组词汇类聚。清末民初时期出现在白话报刊中的异##36,115,291,121</9><10>体成语，便是清末民初异体成语。清末民初异体成语中在当时使用频率最高的变体我们称##36,125,291,130</10><11>为清末民初异体成语的通体，其他称为清末民初异体成语的变体。##36,135,223,140</11><12>清末民初异体成语可以从形式、语义、结构等方面分类分析，总结分布特征。清末民##48,144,291,149</12><13>初异体成语在数量与分布上具有繁复性，在类型上具有多样化特征，在风格色彩上具有白##35,153,290,159</13><14>话化特征，在整体运用上具有过渡性特征等。##36,163,164,168</14><15>43##36,296,44,300</15><16>万方数据##26,310,48,315</16>",
    ]

    for i in range (1):
        batch_texts = batchfy_text(all_texts, args.batch_size)
        for bs, texts in enumerate(batch_texts):
            for i in range(10):
                import datetime
                starttime = datetime.datetime.now()
                
                outputs = predictor.predict(texts)

                endtime = datetime.datetime.now()
                duringtime = endtime - starttime
                print ("耗时:",duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)
            
            for text, result in zip(texts, outputs["result"]):
                print("{}\n{}".format(text, result))

    import datetime
    import time
    starttime = datetime.datetime.now()

    for i in range (1):
        batch_texts = batchfy_text(all_texts, args.batch_size)
        for bs, texts in enumerate(batch_texts):
            outputs = predictor.predict(texts)
            for text, result in zip(texts, outputs["result"]):
                print("{}\n{}".format(text, result))
    
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    print (duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)# 单位是毫秒
    # 104090
