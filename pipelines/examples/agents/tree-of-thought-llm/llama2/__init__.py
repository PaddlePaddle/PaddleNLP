"""
author: Ernestina
des: 1) set configure 2) initiate llama2
"""
import time
import yaml
from typing import List, Optional
from llama2.llama.llama import Llama, Dialog

import os
os.environ["WORLD_SIZE"] = '1'
os.environ["RANK"] = '0'
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '8020'

llm_config_path = os.path.join(os.getcwd(), "llm_config.yml")
with open(llm_config_path, 'r') as f:
    log_config = yaml.full_load(f.read())


class ChatCompletion:
    global log_config
    global max_seq_len
    global max_batch_size
    def __init__(self, model="llama-2-7b-chat") -> None:
        ckpt_dir = log_config[model]["ckpt_dir"]
        tokenizer_path = log_config[model]["tokenizer_path"]
        # ckpt_dir = f"/mnt/e/study/dl/llama2/{model}/"
        # tokenizer_path = "/mnt/e/study/dl/llama2/tokenizer.model"
        max_seq_len = 1000
        max_batch_size = 6
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            )

    # @staticmethod
    def create(
        self,
        messages: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9, max_gen_len: Optional[int] = None):
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
        results = self.generator.chat_completion(
            messages,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        completion = {
                        "choices": [],
                        "created": time.time(),
                        "id": "llama2_{}".format(int(time.time())),
                        "model": "llama-2-7b-chat",
                        "object": "chat.completion",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                            "total_tokens": 0
                        }
                        }
    
        assert len(messages) == len(results)
        for i in range(len(results)):
            dialog = messages[i]
            print(f"dialog: \n {dialog}")
            result = results[i]
            if i == len(results) - 1:
                finish_reason = "stop"
            else:
                finish_reason = "length"
            tmp = {
                    "finish_reason": finish_reason,
                    "index": i,
                    "message": {"content": "", "role": ""}
                    }
            tmp["message"]["role"] = result["generation"]['role']
            tmp['message']['content'] = result['generation']['content'].replace("\n", "")

            completion["choices"].append(tmp)
            print(f"\n result: \n {result}")
            
    
        return completion