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

# a unittest for gpt-3 checkpoint converter
import os
import subprocess
import unittest


class TestConverter(unittest.TestCase):
    def get_pretrain_cmd(self):
        assert self.dp_size * self.tp_size * self.fsdp_size == 8, "dp_size * tp_size * fsdp_size must be 8 (num_gpus)"
        backend_str = " " if self.backend is None else ("--transformer_engine_backend " + self.backend + " ")
        te_init_weight_path_str = (
            " " if self.te_init_weight_path is None else ("--te_init_weight_path " + self.te_init_weight_path + " ")
        )
        cmd = (
            "export PYTHONPATH=/workspace/PaddleNLP:${PYTHONPATH} && "
            + "python -u  -m paddle.distributed.launch "
            + "--gpus 0,1,2,3,4,5,6,7 "
            + "--log_dir logs/tmp "
            + "run_pretrain.py "
            + "--model_type gpt "
            + "--model_name_or_path gpt3-13B-en "
            + "--tokenizer_name_or_path gpt3-13B-en "
            + "--input_dir /workspace/llama_openwebtext "
            + "--output_dir output/tmp "
            + "--split 949,50,1 "
            + "--max_seq_length 1024 "
            + "--per_device_train_batch_size 1 "
            + "--per_device_eval_batch_size 1 "
            + "--tensor_parallel_degree "
            + str(self.tp_size)
            + " "
            + "--sharding_parallel_degree "
            + str(self.fsdp_size)
            + " "
            + "--pipeline_parallel_degree 1 "
            + "--fuse_attention_qkv 1 "
            + "--use_flash_attention 0 "
            + "--bf16 "
            + "--fp16_opt_level O2 "
            + "--scale_loss 1024 "
            + "--learning_rate 0.00001 "
            + "--min_learning_rate 0.000005 "
            + "--max_steps "
            + str(self.max_steps)
            + " "
            + "--save_steps "
            + str(self.save_steps)
            + " "
            + "--weight_decay 0.01 "
            + "--warmup_ratio 0.01 "
            + "--max_grad_norm 1.0 "
            + "--logging_steps 1 "
            + "--dataloader_num_workers 1 "
            + "--hidden_dropout_prob 0.1 "
            + "--attention_probs_dropout_prob 0.1 "
            + "--eval_steps 1000 "
            + "--report_to visualdl "
            + "--disable_tqdm true "
            + "--gradient_accumulation_steps 1 "
            + "--do_train "
            + "--do_eval "
            + "--continue_training 0 "
            + "--sharding stage2 "
            + backend_str
            + "--recompute 1 --recompute_granularity full "
            + te_init_weight_path_str
            + "--device gpu"
        )
        print(f"pretrain cmd: {cmd}")
        return cmd

    def get_convert_cmd(self):
        cmd = (
            "python te_ckpt_converter.py "
            + "--input_ckpt_path "
            + self.input_ckpt_path
            + " "
            + "--output_ckpt_path "
            + self.output_ckpt_path
            + " "
            + "--mode "
            + self.mode
        )
        print(f"converter cmd: {cmd}")
        return cmd

    def check_log(self):
        # check if the log file exists
        self.assertTrue(os.path.exists(self.pretrain_log))
        self.assertTrue(os.path.exists(self.convert_pd2te_log))
        self.assertTrue(os.path.exists(self.convert_te2pd_log))
        self.assertTrue(os.path.exists(self.pretrain_te2pd_log))
        self.assertTrue(os.path.exists(self.pretrain_pd2te_log))

        # check if the log file contains the correct info
        with open(self.pretrain_log, "r") as f:
            self.assertTrue("Saving model checkpoint" in f.read())
        with open(self.convert_pd2te_log, "r") as f:
            self.assertTrue("success" in f.read())
        with open(self.convert_te2pd_log, "r") as f:
            self.assertTrue("success" in f.read())
        with open(self.pretrain_te2pd_log, "r") as f:
            self.assertTrue("Saving model checkpoint" in f.read())
        with open(self.pretrain_pd2te_log, "r") as f:
            self.assertTrue("Saving model checkpoint" in f.read())

        # rm the log file
        os.system("rm -rf logs/tmp*")

    def check_ckpt_converter(self):
        # clear output dir if not empty
        if os.path.exists("output/tmp"):
            os.system("rm -rf output/tmp*")
        # clear log dir if not empty
        if os.path.exists("logs/tmp"):
            os.system("rm -rf logs/tmp*")

        self.backend = None
        self.te_init_weight_path = None
        cmd = self.get_pretrain_cmd()
        # use subprocess to run the command
        output = subprocess.check_output(cmd, shell=True)
        # save output to file
        self.pretrain_log = "logs/tmp/pretrain_output.txt"
        with open(self.pretrain_log, "w") as f:
            f.write(str(output))

        # convert from pd to te
        self.input_ckpt_path = "output/tmp/checkpoint-1"
        self.output_ckpt_path = "output/tmp_converted_ckpt/checkpoint-1-pd2te"
        self.mode = "pd2te"
        # rm the output dir if not empty
        if os.path.exists(self.output_ckpt_path):
            os.system("rm -rf " + self.output_ckpt_path + "/*")
        # mkdir -p the output dir
        os.system("mkdir -p " + self.output_ckpt_path)
        convert_cmd = self.get_convert_cmd()
        # use subprocess to run the command
        output = subprocess.check_output(convert_cmd, shell=True)
        # save output to file
        self.convert_pd2te_log = "logs/tmp/convert_pd2te_output.txt"
        with open(self.convert_pd2te_log, "w") as f:
            f.write(str(output))

        # convert from te to pd
        self.input_ckpt_path = "output/tmp_converted_ckpt/checkpoint-1-pd2te"
        self.output_ckpt_path = "output/tmp_converted_ckpt/checkpoint-1-te2pd"
        self.mode = "te2pd"
        # rm the output dir if not empty
        if os.path.exists(self.output_ckpt_path):
            os.system("rm -rf " + self.output_ckpt_path + "/*")
        # mkdir the output dir
        os.system("mkdir -p " + self.output_ckpt_path)
        convert_cmd = self.get_convert_cmd()
        # use subprocess to run the command
        output = subprocess.check_output(convert_cmd, shell=True)
        # save output to file
        self.convert_te2pd_log = "logs/tmp/convert_te2pd_output.txt"
        with open(self.convert_te2pd_log, "w") as f:
            f.write(str(output))

        # rm the original output dir to avoid auto loading
        os.system("rm -rf output/tmp/*")

        # run the pretrain again with converted pd checkpoint
        self.backend = None
        self.te_init_weight_path = "output/tmp_converted_ckpt/checkpoint-1-te2pd"
        cmd = self.get_pretrain_cmd()
        # use subprocess to run the command
        output = subprocess.check_output(cmd, shell=True)
        # save output to file
        self.pretrain_te2pd_log = "logs/tmp/pretrain_output_te2pd.txt"
        with open(self.pretrain_te2pd_log, "w") as f:
            f.write(str(output))

        # rm the original output dir to avoid auto loading
        os.system("rm -rf output/tmp/*")

        # run the pretrain again with converted te checkpoint
        self.backend = "transformer_engine"
        self.te_init_weight_path = "output/tmp_converted_ckpt/checkpoint-1-pd2te"
        cmd = self.get_pretrain_cmd()
        # use subprocess to run the command
        output = subprocess.check_output(cmd, shell=True)
        # save output to file
        self.pretrain_pd2te_log = "logs/tmp/pretrain_output_pd2te.txt"
        with open(self.pretrain_pd2te_log, "w") as f:
            f.write(str(output))

        # rm the output dir
        os.system("rm -rf output/tmp*")

        # check the log file
        self.check_log()

    def test_check_ckpt_converter_fsdp(self):
        self.dp_size, self.tp_size, self.fsdp_size = 1, 1, 8
        self.max_steps, self.save_steps = 2, 1
        self.check_ckpt_converter()

    def test_check_ckpt_converter_tp(self):
        self.dp_size, self.tp_size, self.fsdp_size = 1, 8, 1
        self.max_steps, self.save_steps = 2, 1
        self.check_ckpt_converter()

    def test_check_ckpt_converter_fsdp_tp(self):
        self.dp_size, self.tp_size, self.fsdp_size = 1, 2, 4
        self.max_steps, self.save_steps = 2, 1
        self.check_ckpt_converter()


if __name__ == "__main__":
    unittest.main()
