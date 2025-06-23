# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3,4,5,6,7 pp.py
import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset
from paddlenlp.experimental.galvatron.runtime.redistributed import SpiltBatchFwdGatherBatchBwd, GatherBatchFwdSplitBatchBwd

mesh1 = dist.ProcessMesh([[0, 1, 2, 3]], dim_names=['dp', 'mp'])
mesh2 = dist.ProcessMesh([[4, 5, 6, 7]], dim_names=['dp', 'mp'])
mesh3 = dist.ProcessMesh([[0], [1], [2], [3]], dim_names=['dp', 'mp'])
mesh4 = dist.ProcessMesh([[4], [5], [6], [7]], dim_names=['dp', 'mp'])
pp_degree = 2

paddle.seed(42)
np.random.seed(42)

class RandomDataset(Dataset):
    def __init__(self, image_size, num_samples=100):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.image_size]).astype("float32")
        label = np.random.uniform(size=[10]).astype("float32")
        return input, label

    def __len__(self):
        return self.num_samples

class DummyLayer(nn.Layer): # 这个纯粹占位符
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class LinearModel(nn.Layer):
    def __init__(self, num_layers, image_size, class_num):
        super().__init__()
        self.num_layers = num_layers
        self.image_size = image_size
        self.class_num = class_num

        self.meshs = [mesh1 for _ in range(4)] + [mesh2 for _ in range(4)]

        self.linears = nn.LayerList()
        
        for i in range(num_layers):
            linear = nn.Linear(image_size, image_size, bias_attr=False)
            if dist.get_rank() in [0, 1, 2, 3]:
                mesh = mesh1
            else:
                mesh = mesh4
            linear.weight = dist.shard_tensor(
                linear.weight,
                # self.meshs[i],
                mesh,
                [dist.Replicate()],
            )
            # self.linears.append(linear) # 原始方案
            if dist.get_rank() in [0, 1, 2, 3]:
                if i < 4: 
                    self.linears.append(linear)
                else:
                    self.linears.append(DummyLayer())
            if dist.get_rank() in [4, 5, 6, 7]:
                if i >= 4:
                    self.linears.append(linear)
                else:
                    self.linears.append(DummyLayer())
            
            
        # self.o_proj = nn.Linear(image_size, class_num, bias_attr=False)
        # self.o_proj.weight = dist.shard_tensor(
        #     self.o_proj.weight,
        #     mesh2,
        #     [dist.Replicate()],
        # )
        
    def forward(self, x):
        x.stop_gradient = False
        out = x
        rank = dist.get_rank()
        
        for i in range(self.num_layers):
            if i == 4:
                out = dist.reshard(out, mesh2, out.placements)
            print(f'rank {rank} layer {i} input: {out}')
            out = self.linears[i](out)
            if i == 3 and rank in [0, 1, 2, 3]:
                out = SpiltBatchFwdGatherBatchBwd.apply(out, mesh3) # 需要转换为mesh3一样的
            
        return out
        print(f'rank {rank} layer before o_proj output: {out}')
        out = self.o_proj(out)
        print(f'rank {rank} layer after o_proj output: {out}')
        return paddle.cast(out, 'float32')

# model = LinearModel(num_layers=8, image_size=4096, class_num=10)

# print(f'model is {model}')

dataset = RandomDataset(4096)
sampler = BatchSampler(
    dataset,
    batch_size=8,
    drop_last=True,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)

rank = dist.get_rank()
if rank in [0, 1, 2, 3]:
    dist_dataloader = dist.shard_dataloader(dataloader, shard_dims=[0, 0], meshes=[mesh1, mesh2])
else:
    dist_dataloader = dist.shard_dataloader(dataloader, shard_dims=[0, 0], meshes=[mesh3, mesh4])

train_dataloader = dist_dataloader()

for step, inputs in enumerate(train_dataloader):
    print(f'rank {rank} step {step} inputs: {inputs}')

# opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())

# def loss_func(output, label):
#     return output.mean()  # 简单的均值损失函数
# # loss_fn = nn.MSELoss()
# loss_fn = loss_func

# # 配置流水并行参数
# strategy = dist.Strategy()
# pipeline = strategy.pipeline
# pipeline.enable = True
# pipeline.schedule_mode = "1F1B"
# pipeline.pp_degree = 2
# pipeline.accumulate_steps = 4

# model = dist.to_static(model, dist_dataloader, loss_fn, opt, strategy)
# model.train()

# rank = dist.get_rank()
# for step, inputs in enumerate(dist_dataloader):
#     loss = model(inputs)
#     print(f'rank {rank} step {step} loss: {loss}')

# print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB") # 671.48 MB
