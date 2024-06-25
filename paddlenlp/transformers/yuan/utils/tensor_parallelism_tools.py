""" Yuan model tools"""

import paddle


print("load yuan weights ......")
model = paddle.load('/workspace/model_state.pdparams')
print("load yuan weights finish ......")
#set tensor parallel degree 
tp_degree = 2
#set the number of hidden_layers, from config.json
hidden_layers = 24
size = model[f"model.layers.0.self_attn.q_proj.weight"].shape[0]
step = size//tp_degree
for i in range(0,hidden_layers):
    
    q = model[f"model.layers.{i}.self_attn.q_proj.weight"]
    k = model[f"model.layers.{i}.self_attn.k_proj.weight"]
    q_slices = [q[:, i:i+step] for i in range(0, size, step)]
    k_slices = [k[:, i:i+step] for i in range(0, size, step)]
    q1=paddle.concat(q_slices[0::2],1)
    q2=paddle.concat(k_slices[0::2],1)
    k1=paddle.concat(q_slices[1::2],1)
    k2=paddle.concat(k_slices[1::2],1)

    model[f"model.layers.{i}.self_attn.q_proj.weight"]=paddle.concat([q1,q2],1)
    model[f"model.layers.{i}.self_attn.k_proj.weight"]=paddle.concat([k1,k2],1)
    print(i," layer is finished ......")

paddle.save(model, '/workspace/yuan_paddle/model_state.pdparams')

