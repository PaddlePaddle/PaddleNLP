# RUN_STATIC=1 python -u -m paddle.distributed.launch --gpus "0,1,2,3" embedding_debug.py
# RUN_STATIC=0 python -u -m paddle.distributed.launch --gpus "0,1,2,3" embedding_debug.py
import os
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset

mesh0 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])

paddle.seed(1024)
np.random.seed(1024)
class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=8): # 序列长度为4，隐藏维度为8
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples
        self.inputs = [np.random.uniform(size=[self.seq_len, self.hidden]).astype("int64") for _ in range(num_samples)]
        self.labels = [np.array(index, dtype="int64") for index in range(num_samples)]

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return self.num_samples

class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.embedding = paddle.nn.Embedding(1024, 4096)
        self.embedding.weight = dist.shard_tensor(
            self.embedding.weight,
            mesh0, [dist.Replicate(), dist.Shard(1)])
        # self.w0 = dist.shard_tensor(
        #     self.create_parameter(shape=[8, 8]),
        #     mesh0, [dist.Replicate(), dist.Shard(1)])

    def forward(self, x):
        y = self.embedding(x)
        y = dist.reshard(y, mesh0, [dist.Shard(1), dist.Shard(0)])
        # y = hidden_states = dist.reshard(hidden_states, get_mesh(), self.placements)
        return y
        # y = paddle.matmul(x, self.w0)
        # return y

model = MlpModel()
dataset = RandomDataset(4, 8)
sampler = BatchSampler(
    dataset,
    batch_size=2,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)
dist_dataloader = dist.shard_dataloader(
    dataloader=dataloader,
    meshes=mesh0,
    shard_dims="x"
)
opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
dist_opt = dist.shard_optimizer(opt, dist.ShardingStage3("x", mesh0))

def loss_fn(output, label):
    return output
# def loss_fn(logits, label):
#     # logits: [bs, seq_len, hidden], label: [bs]
#     loss = paddle.nn.MSELoss(reduction="sum")
#     logits = paddle.sum(logits, axis=[1, 2])
#     return loss(logits, label)

RUN_STATIC = eval(os.environ['RUN_STATIC'])
def run_dynamic():
    for step, (input, label) in enumerate(dist_dataloader()):
        print(f'input: {input}, label: {label}')
        logits = model(input)
        print("logits:", logits)
        
        loss = loss_fn(logits, label)
        print("step:{}, loss:{}".format(step, loss))
        loss.backward()
        dist_opt.step()
        dist_opt.clear_grad()

def run_static():
    dist_model = dist.to_static(
        model, dist_dataloader, loss_fn, opt
    )
    dist_model.train()
    for step, (input, label) in enumerate(dist_dataloader()):
        print("label:", label)
        loss = dist_model(input, label)
        print("step:{}, loss:{}".format(step, loss))

if RUN_STATIC == 0:
    run_dynamic()
else:
    run_static()

# This case need to be executed in multi-card environment
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# RUN_STATIC=1 python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" embedding_debug.py
# RUN_STATIC=0 python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" embedding_debug.py
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# RUN_STATIC=1 python -u -m paddle.distributed.launch --gpus "0,1,2,3" embedding_debug.py
# RUN_STATIC=0 python -u -m paddle.distributed.launch --gpus "0,1,2,3" embedding_debug.py

# for _ in range(5):
#     loss = layer(batch)
#     loss.backward()
#     opt.step()
#     opt.clear_grad()

# # python -m paddle.distributed.launch --gpus=0,1 embedding_debug.py
# import paddle
# import paddle.distributed as dist
# mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
# class MLP(paddle.nn.Layer):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = paddle.nn.Linear(1024, 1024)
#         self.fc2 = paddle.nn.Linear(1024, 1024)
#         self.fc1.weight = dist.shard_tensor(self.fc1.weight, mesh, [dist.Replicate()])
#         self.fc2.weight = dist.shard_tensor(self.fc2.weight, mesh, [dist.Replicate()])

#     def forward(self, input):
#         return self.fc2(self.fc1(input))
    
# layer = MLP()
# print(layer)

# batch = paddle.rand(shape=[1024, 1024])
# batch = dist.shard_tensor(batch, mesh, [dist.Shard(0)])

# dataset = RandomDataset(4, 8)
# sampler = BatchSampler(
#     dataset,
#     batch_size=2,
# )
# dataloader = DataLoader(
#     dataset,
#     batch_sampler=sampler,
# )

# opt = paddle.optimizer.AdamW(parameters=layer.parameters())
# opt = dist.shard_optimizer(opt, dist.ShardingStage3("x", mesh))
# print("after shard_optimizer layer is ")
# print(layer)

# model = dist.to_static(layer=layer,optimizer=opt)
# print("after to_static layer is ")
# print(model)

# for _ in range(5):
#     loss = model(batch)
#     loss.backward()
#     opt.step()
#     opt.clear_grad()
    
    