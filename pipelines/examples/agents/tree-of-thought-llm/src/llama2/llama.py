#coding=utf8, ErnestinaQiu
import os
import time
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM


llm_config = {
              "llama-2-7b": "meta-llama/Llama-2-7b", "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat",
              "llama-2-13b": "meta-llama/Llama-2-13b", "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat",
              "llama-2-70b": "meta-llama/Llama-2-70b", "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat"
              }


class llamaChatCompletion:
    global llm_config

    def __init__(self, model="llama-2-7b-chat") -> None:
        config_path = llm_config[model]
        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        self.generator = AutoModelForCausalLM.from_pretrained(config_path, dtype="float16")

    # @staticmethod
    def create(self, messages, temperature=0.6, top_p=0.9, max_gen_len=518):
        """
        Entry point of the program for generating text using a pretrained model.

        Args:
            messages (list): There are two roles including "system" and "user".
            --Example  [[{"role": "user", "content": "what is the recipe of mayonnaise?"}, {"role": "system", "content": "Always answer with Haiku"}]]
            ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
            tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
            temperature (float, optional): The temperature value for controlling randomness in generation.
                Defaults to 0.6.
            top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
                Defaults to 0.9.
            max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
            max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
            max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
                set to the model's max sequence length. Defaults to None.
        """
        completion = {
            "choices": [],
            "created": time.time(),
            "id": "llama2_{}".format(int(time.time())),
            "model": "llama-2-7b-chat",
            "object": "chat.completion",
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }

        for i in range(len(messages)):
            one_mes = messages[i][0]
            assert len(messages[i]) == 1
            if one_mes["role"] != "user":
                continue
            mes = one_mes["content"]
            input_features = self.tokenizer(mes, return_tensors="pd")
            outputs = self.generator.generate(**input_features, max_new_tokens=max_gen_len)
            out_0 = self.tokenizer.batch_decode(outputs[0])
            print(f"dialog: \n {one_mes}")
            print(out_0)
            if i == len(messages) - 1:
                finish_reason = "stop"
            else:
                finish_reason = "length"
            tmp = {
                "finish_reason": finish_reason,
                "index": i,
                "message": {"content": "", "role": ""},
            }
            tmp["message"]["role"] = "llm"
            tmp["message"]["content"] = out_0
            completion["choices"].append(tmp)

        return completion
