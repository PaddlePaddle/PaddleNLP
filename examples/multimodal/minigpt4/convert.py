import torch
import numpy as np
vit_model = torch.load("chinese_fenpian_epoch0_step10w.bin")


vit_dict = {}
for k, v in  vit_model.items():
    print(k)
    print(v.dtype)
    print(v.shape)
    vit_dict[k] = v.cpu().numpy()

np.save("vit1.npy", vit_dict, allow_pickle=True))




