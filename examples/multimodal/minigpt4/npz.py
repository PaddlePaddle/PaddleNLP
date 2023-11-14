import numpy as np
import torch

path_data = "/root/paddlejob/workspace/env_run/zhengshifeng/LAVIS_to_onnx/models_bk/test_data/input_and_output_tuwensou_data_baichuan_0epoch.npz"
data = np.load(path_data, allow_pickle=True)

aaa = []

inputs = data["input"]
outputs = data["output"]
for i in range(len(inputs)):
    image = inputs[i]["image"].numpy()
    aaa.append(image)


data = np.save("vit_numpy.npy", aaa)

