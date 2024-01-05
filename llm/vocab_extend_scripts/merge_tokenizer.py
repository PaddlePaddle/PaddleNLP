# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse
import operator
import pickle

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model

from paddlenlp.transformers import AutoTokenizer, LlamaTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin_tokenizer_dir", default=None, required=True, type=str, help="The directory of origin tokenizer model"
    )
    parser.add_argument(
        "--chinese_sp_model_file", default=None, required=True, type=str, help="The directory of new tokenizer model"
    )
    parser.add_argument(
        "--pretrain_files_dir",
        default=None,
        required=True,
        help="The directory of pretrain data,used to tailor the vocab.",
    )
    parser.add_argument(
        "--chinese_sp_vocab_file", default=None, required=True, help="The directory of new *.vocab file."
    )
    parser.add_argument("--output_dir", default=None, required=True, help="The directory of the output.")
    parser.add_argument("--need_to_use_8_gpus_tp", default=False, required=True, help="The directory of the output.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    origin_tokenizer_dir = args.origin_tokenizer_dir
    chinese_sp_model_file = args.chinese_sp_model_file
    # load
    origin_tokenizer = AutoTokenizer.from_pretrained(origin_tokenizer_dir)
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file)

    origin_spm = sp_pb2_model.ModelProto()
    origin_spm.ParseFromString(origin_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    # print number of tokens
    print(len(origin_tokenizer), len(chinese_sp_model))
    print(origin_tokenizer.all_special_tokens)
    print(origin_tokenizer.all_special_ids)
    print(origin_tokenizer.special_tokens_map)

    # Add Chinese tokens to LLaMA tokenizer
    origin_spm_tokens_set = set(p.piece for p in origin_spm.pieces)
    print(len(origin_spm_tokens_set))
    print(f"Before:{len(origin_spm_tokens_set)}")
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in origin_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            origin_spm.pieces.append(new_p)
    print(f"New model pieces: {len(origin_spm.pieces)}")
    # 判断词表大小是否被8整除
    if args.need_to_use_8_gpus_tp and (len(origin_spm.pieces) % 8) != 0:
        print(
            "The size of the new model is not divisible by 8 and it needs to be tailored.Please wait for a few more hours."
        )
        print(
            "The size of the new model is not divisible by 8 and it needs to be tailored.Please wait for a few more hours."
        )
        print(
            "The size of the new model is not divisible by 8 and it needs to be tailored.Please wait for a few more hours."
        )
        print(len(origin_spm.pieces))
        sp = spm.SentencePieceProcessor()
        sp.load(chinese_sp_model_file)
        token_dict = {}
        with open(os.path.join(args.pretrain_files_dir, "all_pretrain_data.txt"), "r") as f:
            lines = f.readlines()
        for line in lines:
            pieces = sp.encode_as_pieces(line)
            for piece in pieces:
                if piece not in token_dict:
                    token_dict[piece] = 1
                else:
                    token_dict[piece] = token_dict[piece] + 1
        with open("test.pkl", "wb") as f:
            pickle.dump(token_dict, f)
        with open("test.pkl", "rb") as f:
            token_dict = pickle.load(f)
        sorted_tokens = sorted(token_dict.items(), key=operator.itemgetter(1))
        with open(args.chinese_sp_vocab_file, "r") as f:
            lines = f.readlines()
            vocab = []
            for line in lines:
                vocab.append(line.split("\t")[0])
        l = 0
        r = len(sorted_tokens)
        while l <= r:
            mid = int((l + r) / 2)
            if sorted_tokens[mid][0] not in vocab:
                l = mid + 1
            else:
                r = mid - 1
        num_tokens_need_to_be_tailored = len(origin_spm.pieces) % 8
        tokens_need_to_be_tailored = []
        for i in range(mid, mid + num_tokens_need_to_be_tailored):
            tokens_need_to_be_tailored.append(sorted_tokens[i][0])
        for item in origin_spm.pieces:
            if item.piece in tokens_need_to_be_tailored:
                origin_spm.pieces.remove(item)
    print(f"New model pieces: {len(origin_spm.pieces)}")
    # Save
    output_sp_dir = os.path.join(args.output_dir, "merged_tokenizer_sp")
    output_paddle_dir = os.path.join(
        args.output_dir, "merged_tokenizer_paddle"
    )  # the path to save Chinese-LLaMA tokenizer
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + "/chinese_llama.model", "wb") as f:
        f.write(origin_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/chinese_llama.model")

    tokenizer.save_pretrained(output_paddle_dir)
    print(f"Chinese-LLaMA tokenizer has been saved to {output_paddle_dir}")

    # Test
    origin_tokenizer = AutoTokenizer.from_pretrained(origin_tokenizer_dir)
    chinese_llama_tokenizer = AutoTokenizer.from_pretrained(output_paddle_dir)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    text = """白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
    The primary use of LLaMA is research on large language models, including"""
    print("Test text:\n", text)
    print(f"Tokenized by LLaMA tokenizer:{origin_tokenizer.tokenize(text)}")
    print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")


if __name__ == "__main__":
    main()
