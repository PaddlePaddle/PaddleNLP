import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
class MoRALinear(nn.Layer):
    """MoRA (Type 6) 的Paddle实现"""
    def __init__(
        self,
        base_layer: nn.Layer,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        
        # 获取输入输出维度
        if isinstance(base_layer, nn.Linear):
            self.in_features = base_layer.weight.shape[0]
            self.out_features = base_layer.weight.shape[1]
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
            
        # Dropout层
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            
        # 计算MoRA的高秩 = [(d_in + d_out) * r]的平方根
        new_r = int(math.sqrt((self.in_features + self.out_features) * r) + 0.5)
        # Type 6需要确保r是偶数(RoPE需要)
        new_r = new_r // 2 * 2
        self.r = new_r
        
        # 创建A矩阵(MoRA中只需要一个权重矩阵)
        self.lora_A = self.create_parameter(
            shape=[new_r, new_r],
            dtype=base_layer.weight.dtype
        )
        
        # 初始化为0
        nn.initializer.Constant(value=0.0)(self.lora_A)
        
        # 缩放因子
        self.scaling = 1.0
        
        # RoPE相关缓存
        self.cos = None
        self.sin = None
        
        # 添加合并状态标志
        self.merged = False
        
    def _apply_mora(self, x):
        """应用MoRA变换"""
        r = self.r
        
        # 计算分组
        sum_inter = self.in_features // r
        rb1 = self.in_features // r if self.in_features % r == 0 else self.in_features // r + 1
        
        # 处理需要padding的情况
        if self.in_features % r != 0:
            pad_size = r - self.in_features % r
            x = paddle.concat([x, x[..., :pad_size]], axis=-1)
            sum_inter += 1
            
        # 重塑输入以准备应用RoPE
        in_x = x.reshape([*x.shape[:-1], sum_inter, r])
        
        # 生成RoPE位置编码
        if self.cos is None or self.sin is None:
            inv_freq = 1.0 / (10000 ** (paddle.arange(0, r, 2, dtype='float16') / r))
            t = paddle.arange(rb1, dtype='float16')
            freqs = paddle.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
            emb = paddle.concat([freqs, freqs], axis=-1)
            self.cos = paddle.unsqueeze(paddle.cos(emb), axis=0).astype(x.dtype)
            self.sin = paddle.unsqueeze(paddle.sin(emb), axis=0).astype(x.dtype)
        
        # 应用RoPE旋转
        rh_in_x = paddle.concat([-in_x[..., r//2:], in_x[..., :r//2]], axis=-1)
        in_x = in_x * self.cos + rh_in_x * self.sin
        
        # 通过MoRA权重矩阵
        out_x = paddle.matmul(in_x, self.lora_A)
        
        # 调整输出维度
        out_x = out_x.reshape([*x.shape[:-1], -1])[..., :self.out_features]
        if out_x.shape[-1] < self.out_features:
            repeat_time = self.out_features // out_x.shape[-1]
            if self.out_features % out_x.shape[-1] != 0:
                repeat_time += 1
            out_x = paddle.concat([out_x] * repeat_time, axis=-1)[..., :self.out_features]
        
        return out_x * self.scaling

    def get_delta_weight(self):
        """计算delta权重矩阵，用于合并权重"""
        r = self.r
        
        # 计算padding
        pad_size = r - self.in_features % r if self.in_features % r != 0 else 0
        
        # 初始化权重矩阵
        w = paddle.zeros([self.in_features + pad_size, self.in_features], dtype=self.lora_A.dtype)
        
        # 计算分块数
        rb1 = self.in_features // r if self.in_features % r == 0 else self.in_features // r + 1
        rb2 = self.out_features // r if self.out_features % r == 0 else self.out_features // r + 1
        
        # 生成RoPE位置编码
        if self.cos is None or self.sin is None:
            inv_freq = 1.0 / (10000 ** (paddle.arange(0, r, 2, dtype='float16') / r))
            t = paddle.arange(rb1, dtype='float16')
            freqs = paddle.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
            emb = paddle.concat([freqs, freqs], axis=-1)
            self.cos = paddle.unsqueeze(paddle.cos(emb), axis=0).astype(self.lora_A.dtype)
            self.sin = paddle.unsqueeze(paddle.sin(emb), axis=0).astype(self.lora_A.dtype)
        
        # 生成旋转后的权重矩阵
        aw2 = paddle.concat([self.lora_A[:, r//2:], -self.lora_A[:, :r//2]], axis=1)
        
        # 对每个完整的分块应用RoPE
        for i in range(rb1 - 1):
            w[i*r:(i+1)*r, i*r:(i+1)*r] = (
                aw2 * self.sin[:, i] + self.lora_A * self.cos[:, i]
            )
        
        # 处理最后一个可能不完整的分块
        i = rb1 - 1
        w[i*r:, i*r:] = (
            aw2 * self.sin[:, i] + self.lora_A * self.cos[:, i]
        )[:, :r-pad_size]
        
        # 如果有填充,处理填充部分
        if pad_size > 0:
            w[i*r:, :pad_size] = (
                aw2 * self.sin[:, i] + self.lora_A * self.cos[:, i]
            )[:, r-pad_size:]
        
        # 调整输出维度
        if self.in_features < self.out_features:
            w = paddle.concat([w] * rb2, axis=0)[:self.out_features]
        else:
            w = w[:self.out_features]
            
        # 修改最后的返回值，确保维度匹配
        final_weight = w * self.scaling
        # 转置权重矩阵以匹配 Linear 层的权重格式
        return final_weight.T

    def merge(self):
        """合并MoRA权重到基础层"""
        if self.merged:
            return
        
        delta_weight = self.get_delta_weight()
        self.base_layer.weight.data += delta_weight
        self.merged = True

    def unmerge(self):
        """从基础层分离MoRA权重"""
        if not self.merged:
            return
            
        delta_weight = self.get_delta_weight()
        self.base_layer.weight.data -= delta_weight
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base_layer(x)
        
        # 基础层输出
        base_out = self.base_layer(x)
        
        # 应用dropout
        x = self.lora_dropout(x)
        
        # 应用MoRA变换
        mora_out = self._apply_mora(x)
        
        return base_out + mora_out

def create_mora_model(base_model, target_modules, r=8, lora_alpha=1, lora_dropout=0.1):
    """将基础模型转换为MoRA模型"""
    trans = {}
    for name, module in base_model.named_sublayers():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                new_module = MoRALinear(
                    module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                trans[name] = new_module

    for name, new_module in trans.items():
        setattr(base_model, name, new_module)
    return base_model


def train_model(
        model,
        train_dataset,
        num_epochs,
        learning_rate=3e-4,
        weight_decay=0.01,
        save_steps=1000,
        save_path="checkpoints",
        gradient_accumulation_steps=1
):
    optimizer = paddle.optimizer.SGD(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        weight_decay=weight_decay,
    )

    num_training_steps = len(train_dataset) * num_epochs
    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=learning_rate,
        T_max=num_training_steps
    )

    model.train()
    global_step = 0
    accumulated_loss = 0
    print('train_dataloader:', train_dataset)
    train_loss = []
    train_acc = []
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataset):
            #print("batch_type:", batch.type())
            outputs = model(**batch)
            #print(f"outputs:{outputs}")
            loss = outputs[0]  # outputs[0] 是 loss
            train_loss.append(loss.item())
            # 累积损失
            loss = loss / gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            # 每到达梯度累积步数，更新一次参数
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                print(f"Epoch: {epoch}, Step: {global_step}, Loss: {accumulated_loss}, total_step: {num_training_steps}")
                accumulated_loss = 0
                global_step += 1

            # 每 `save_steps` 保存一次模型
            if global_step % save_steps == 0:
                paddle.save(
                    {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': lr_scheduler.state_dict(),
                        'global_step': global_step,
                    },
                    f"{save_path}/checkpoint-{global_step}.pdparams"
                )
                with open("./train_loss.txt", 'w') as train_los:
                    train_los.write(str(train_loss))
                '''
                merge_and_eval(model)
                with paddle.no_grad():
                    query = "Please answer the following question with true or false, question: is zip code the same as post code?\n\nAnswer format: true/false"
                    inputs = tokenizer(query, return_tensors="pd")
                    generate_ids = model.generate(
                        **inputs,
                        do_sample=True,
                        max_new_tokens=2048,
                        top_k=10,
                        top_p=0.85,
                        temperature=1,
                        repetition_penalty=1.15,
                        eos_token_id=2,
                        bos_token_id=1,
                        pad_token_id=0,
                    )[0]
                    print(generate_ids)
                    response = tokenizer.batch_decode(generate_ids)[0]
                    print(response)
                # 3. 如果需要继续训练
                unmerge_and_train(model)
                '''
# 添加用于推理的辅助函数
def merge_and_eval(model):
    """合并权重并切换到评估模式"""
    model.eval()
    
    # 合并所有MoRA层的权重
    for name, module in model.named_sublayers():
        if isinstance(module, MoRALinear):
            module.merge()
            
def unmerge_and_train(model):
    """分离权重并切换到训练模式"""
    model.train()
    
    # 分离所有MoRA层的权重
    for name, module in model.named_sublayers():
        if isinstance(module, MoRALinear):
            module.unmerge()


def preprocess_function(example):
    MAX_LENGTH = 256
    # 将问题和背景信息组合在一起作为输入
    input_text = f"<s>{example['instruction']}\n"
    output_text = f"{example['output']}\n"
    answer_text = f"{example['answer']}</s>"  # 答案部分
    # 编码输入内容
    input_encoding = tokenizer(
        input_text,
        add_special_tokens=False,
    )
    #print(f"input_encoding:{input_encoding}")
    #print(f"解码后的输入：{tokenizer.decode(input_encoding.input_ids)}")
    output_encoding = tokenizer(
        output_text,
        add_special_tokens=False,
    )

    # 编码答案内容，用作 labels
    answer_encoding = tokenizer(
        answer_text,
        add_special_tokens=False,
    )
    #print(f"解码后的输出：{tokenizer.decode(answer_encoding.input_ids)}")
    # 创建包含 input_ids、attention_mask 和 labels 的字典
    input_ids = input_encoding["input_ids"] + output_encoding["input_ids"] + answer_encoding["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = input_encoding["attention_mask"] + output_encoding["attention_mask"] + answer_encoding['attention_mask'] + [0]
    #print(f"attention_mask: {attention_mask}")
    # 将 labels 设置为 answer_encoding，仅包含 Answer 的内容
    labels = [-100] * len(input_encoding["input_ids"]) + output_encoding["input_ids"] + answer_encoding["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100  # 将填充位置设为 -100

    #print(f"input_ids: {input_ids}")
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    if(len(input_ids) < MAX_LENGTH):
        input_ids = input_ids + (MAX_LENGTH - len(input_ids))*[tokenizer.pad_token_id]
        attention_mask = attention_mask + (MAX_LENGTH - len(attention_mask))*[0]
        labels = (MAX_LENGTH - len(labels))*[-100] + labels
    

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    


class TextDataset(paddle.io.Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __getitem__(self, idx):
        #print(f"input_ids:\n{self.data[idx]['input_ids']}")
        return {
            'input_ids': paddle.to_tensor(self.data[idx]['input_ids']),
            'attention_mask': paddle.to_tensor(self.data[idx]['attention_mask']),
            'labels': paddle.to_tensor(self.data[idx]['labels'])
        }

    def __len__(self):
        return len(self.data)


# 使用示例
if __name__ == "__main__":
    # 1. 加载模型和tokenizer
    from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    model_path = "facebook/llama-7b"  # 替换为你的模型路径
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_flash_attention=True,
        dtype="float16"  # 使用float16减少显存占用
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    #tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = '<pad>'
    print("pad_token_id:", tokenizer.pad_token_id)
    print("vocab_size after resizing:", model.config.vocab_size)
    
    #加载数据集
    d_name = "commonsense_170k/train.json"
    dt = load_dataset('json', data_files=d_name)['train']
    tokenizer_id = dt.map(preprocess_function)
    tokenizer_id = tokenizer_id.remove_columns(['instruction', 'input', 'output', 'answer'])
    train_dataset = TextDataset(tokenizer_id)
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_size=16,  # 设置你需要的batch大小
        shuffle=False,
        num_workers=0,  # 根据需要调整数据加载的线程数
    )
    '''
    def collate_fn(batch):
        # 找到 batch 中最长的序列长度
        max_len = 256

        # 对 'input_ids'、'attention_mask' 和 'labels' 进行填充
        input_ids = [F.pad(item['input_ids'], [0, max_len - len(item['input_ids'])], value=tokenizer.pad_token_id) for item
                     in batch]
        attention_mask = [
            F.pad(item['attention_mask'], [0, max_len - len(item['attention_mask'])], value=0) for
            item in batch]
        labels = [F.pad(item['labels'], [0, max_len - len(item['labels'])], value=-100) for item in
                  batch]  # 填充 -100 以忽略损失

        # 将每个字段的列表堆叠成 batch 形式
        return {
            'input_ids': paddle.stack(input_ids),
            'attention_mask': paddle.stack(attention_mask),
            'labels': paddle.stack(labels)
        }
        
    
    
    '''
    # 2. 应用MoRA
    target_modules = [
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "down_proj", 
        "up_proj"
    ]
    
    model = create_mora_model(
        model,
        target_modules=target_modules,
        r=8,
        lora_alpha=1,
        lora_dropout=0.1,
    )
    
    # 4. 训练模型
    train_model(
        model,
        train_dataloader,
        num_epochs=1,
        learning_rate=3e-4,
        save_steps=1000,
        save_path="mora_checkpoints"
    )
    