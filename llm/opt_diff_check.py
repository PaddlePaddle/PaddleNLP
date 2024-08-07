from safetensors import safe_open
import argparse
import numpy as np
import paddle
import json
import os

class opt_handler:
    def __init__(self, index_file, ckpt_path):
        self.info = {}
        self.ckpt_path = ckpt_path
        self.file_io_map = {}
        with open(index_file, 'r') as f:
            self.info = json.load(f)
    
    def get_fd(self, key):
        file_name = self.info["weight_map"][key]
        if file_name not in self.file_io_map.keys():
            self.file_io_map[file_name] = safe_open(os.path.join(self.ckpt_path, file_name), framework="np", device="cpu")
        return self.file_io_map[file_name]

    def get_tensors(self, key):
        return self.get_fd(key).get_tensor(key)

def main(args):
    mmax_diff = 0.0
    mmax_diff_key = None
    vmax_diff = 0.0
    vmax_diff_key = None
    umax_diff = 0.0
    umax_diff_key = None
    wmax_diff = 0.0
    wmax_diff_key = None
    shard_num = 4
    shard_name = "00004"
    for i in range(shard_num):
        eps = 1e-8
        handler1 = opt_handler(os.path.join(args.checkpoint_path1, "optimizer.safetensors.index.json"), args.checkpoint_path1)
        handler2 = opt_handler(os.path.join(args.checkpoint_path2, "optimizer.safetensors.index.json"), args.checkpoint_path2)
        path = os.path.join(args.checkpoint_path1, f"model-0000{i+1}-of-{shard_name}.safetensors")
        path1 = os.path.join(args.checkpoint_path2, f"model-0000{i+1}-of-{shard_name}.safetensors")
        with safe_open(path, framework='np') as weight, safe_open(path1, framework='np') as weight1:
            print(f"checking model-0000{i+1}-of-{shard_name}.safetensors!")
            for key in weight.keys():
                k_m = key + '/moment1_0'
                k_v = key + '/moment2_0'
                w1 = paddle.Tensor(weight.get_tensor(key), zero_copy=True)
                w2 = paddle.Tensor(weight1.get_tensor(key), zero_copy=True)
                opt_m1 = handler1.get_tensors(k_m)
                opt_v1 = handler1.get_tensors(k_v)
                opt_m2 = handler2.get_tensors(k_m)
                opt_v2 = handler2.get_tensors(k_v)
                update_radio1 = np.abs(opt_m1 / (np.sqrt(opt_v1) + eps)).max()
                update_radio2 = np.abs(opt_m2 / (np.sqrt(opt_v2) + eps)).max()
                diff_m = np.abs(opt_m1 - opt_m2).max()
                diff_v = np.abs(opt_v1 - opt_v2).max()
                diff_update_radio = np.abs(update_radio1 - update_radio2).max()
                diff_w = (w1 - w2).abs().max()

                if diff_w > wmax_diff:
                    wmax_diff = diff_w
                    wmax_diff_key = key

                print(f"ratio detect {diff_update_radio}")
                if diff_update_radio > umax_diff:
                    umax_diff = diff_update_radio
                    umax_diff_key = [k_m, k_v]
                    if diff_update_radio > 7:
                        print(f"broken ratio detect {diff_update_radio}")

                if diff_m > mmax_diff:
                    mmax_diff = diff_m
                    mmax_diff_key = k_m

                if diff_v > vmax_diff:
                    vmax_diff = diff_v
                    vmax_diff_key = k_v
                #print(f"{key} has max diff {diff_m} and {diff_v}")
    print("=============================")
    print(f"moment1 diff: {mmax_diff_key} has max diff {mmax_diff}")
    print(f"moment2 diff: {vmax_diff_key} has max diff {vmax_diff}")
    print(f"ratio diff: {umax_diff_key} has max diff {umax_diff}")
    print(f"model weight diff: {wmax_diff_key} has max diff {wmax_diff}")
    print("=============================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test a SAM')
    parser.add_argument('checkpoint_path1', type=str, default="checkpoints/ckpt_quant_pt1/checkpoint-500")
    parser.add_argument('checkpoint_path2', type=str, default="checkpoints/ckpt_quant_pt1/checkpoint-500-ori")
    args, _ = parser.parse_known_args()
    main(args)