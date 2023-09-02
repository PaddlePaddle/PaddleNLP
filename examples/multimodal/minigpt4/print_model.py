import paddle
a = paddle.load("/root/paddlejob/workspace/env_run/zhengshifeng/vitllm/vit_model/model_state.pdparams")
for k, v in a.items():
    print(k,v.shape)
