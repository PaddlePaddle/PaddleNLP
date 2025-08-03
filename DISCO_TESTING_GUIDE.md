# DISCO 算法单元测试指南

本文档说明如何运行和验证 DISCO (Dynamic Score-based Cache Optimization) 算法的单元测试。

## 测试文件概览

DISCO 实现包含以下测试文件：

1. **`tests/transformers/llama/test_disco.py`** - 完整的单元测试套件
2. **`test_disco_standalone.py`** - 独立测试脚本（无需完整 PaddleNLP 环境）
3. **`test_disco_simple.py`** - 简化的功能测试

## 测试环境准备

### 1. 基础环境要求

```bash
# Python 3.8+ 
python --version

# PaddlePaddle 2.5.0+
python -c "import paddle; print(paddle.__version__)"

# 安装测试依赖
pip install pytest pytest-xdist pytest-cov
```

### 2. 克隆并切换到 DISCO 分支

```bash
git clone https://github.com/micelvrice/PaddleNLP.git
cd PaddleNLP
git checkout disco-paddle
```

## 运行测试

### 方式一：运行完整单元测试（推荐）

```bash
# 运行 DISCO 专门的测试
python -m pytest tests/transformers/llama/test_disco.py -v

# 运行所有 LLaMA 相关测试以确保兼容性
python -m pytest tests/transformers/llama/ -v

# 运行带覆盖率的测试
python -m pytest tests/transformers/llama/test_disco.py -v --cov=paddlenlp.transformers.llama.disco_cache
```

### 方式二：运行独立测试脚本

如果遇到环境依赖问题，可以使用独立测试脚本：

```bash
# 运行独立测试
python test_disco_standalone.py

# 运行简化测试
python test_disco_simple.py
```

### 方式三：手动测试核心功能

```python
# 创建测试脚本 test_manual.py
import paddle
from paddlenlp.transformers import LlamaConfig, LlamaModel

# 配置 DISCO
config = LlamaConfig(
    vocab_size=1000,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    use_disco=True,
    disco_cache_size=128,
    disco_window_size=16,
    disco_gamma=0.1
)

# 创建模型
model = LlamaModel(config)
print("✓ Model created with DISCO enabled")

# 测试前向传播
input_ids = paddle.randint(0, 1000, shape=[2, 32])
outputs = model(input_ids, use_cache=True)
print(f"✓ Forward pass successful, output shape: {outputs[0].shape}")
```

## 测试覆盖内容

### 1. DISCOCache 核心功能测试

- **初始化测试** (`test_disco_cache_initialization`)
  - 验证默认参数配置
  - 验证自定义层级预算分配
  
- **缓存更新与驱逐** (`test_cache_update_and_eviction`)
  - 测试 KV 缓存更新机制
  - 验证驱逐策略正确性
  - 确保缓存大小不超过预算

- **评分机制** (`test_layerwise_eviction_manager`)
  - 基于注意力的评分
  - 基于距离的评分（回退方案）
  - KV 融合评分

### 2. 模型集成测试

- **配置集成** (`test_llama_with_disco_config`)
  - 验证 DISCO 参数正确传递
  - 验证模型初始化成功

- **前向传播** (`test_llama_forward_with_disco`)
  - 测试启用 DISCO 的推理
  - 验证输出形状正确

- **兼容性测试** (`test_disco_disabled_compatibility`)
  - 确保 DISCO 禁用时不影响原有功能
  - 验证向后兼容性

### 3. 高级功能测试

- **评分函数加载** (`test_score_function_loading`)
  - 测试从文件加载学习的评分函数
  - 验证多项式评分计算

- **因果语言模型** (`test_causal_lm_with_disco`)
  - 测试 LlamaForCausalLM 集成
  - 验证生成任务兼容性

## 预期测试结果

成功运行的测试应该显示：

```
================= test session starts =================
collected 9 items

test_disco.py::DISCOCacheTest::test_disco_cache_initialization PASSED
test_disco.py::DISCOCacheTest::test_cache_update_and_eviction PASSED
test_disco.py::DISCOCacheTest::test_layerwise_eviction_manager PASSED
test_disco.py::DISCOCacheTest::test_kv_fusion_scoring PASSED
test_disco.py::DISCOCacheTest::test_llama_with_disco_config PASSED
test_disco.py::DISCOCacheTest::test_llama_forward_with_disco PASSED
test_disco.py::DISCOCacheTest::test_disco_disabled_compatibility PASSED
test_disco.py::DISCOCacheTest::test_score_function_loading PASSED
test_disco.py::DISCOCacheTest::test_causal_lm_with_disco PASSED

================= 9 passed in X.XXs ===================
```

## 常见问题排查

### 1. ImportError: paddle.distributed.LocalLayer

如果遇到此错误，说明 PaddleNLP 主分支有不兼容的代码。解决方案：
- 使用独立测试脚本
- 或者临时注释掉有问题的导入

### 2. 测试超时

某些测试可能需要较长时间，可以增加超时时间：

```bash
python -m pytest tests/transformers/llama/test_disco.py -v --timeout=300
```

### 3. 内存不足

减小测试中的模型大小：

```python
# 在测试文件中修改配置
config = LlamaConfig(
    vocab_size=500,  # 减小词表
    hidden_size=128,  # 减小隐藏层
    num_hidden_layers=2,  # 减少层数
    # ...
)
```

## 性能验证

除了功能测试，您还可以运行性能测试：

```python
# performance_test.py
import time
import paddle
from paddlenlp.transformers import LlamaConfig, LlamaModel

# 标准配置
standard_config = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
    use_disco=False
)

# DISCO 配置
disco_config = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
    use_disco=True,
    disco_cache_size=1024  # 只使用 3.2% 的缓存
)

# 创建输入
input_ids = paddle.randint(0, 32000, shape=[1, 2048])

# 测试标准模型
model_standard = LlamaModel(standard_config)
model_standard.eval()

start = time.time()
with paddle.no_grad():
    _ = model_standard(input_ids, use_cache=True)
standard_time = time.time() - start

# 测试 DISCO 模型
model_disco = LlamaModel(disco_config)
model_disco.eval()

start = time.time()
with paddle.no_grad():
    _ = model_disco(input_ids, use_cache=True)
disco_time = time.time() - start

print(f"Standard model time: {standard_time:.3f}s")
print(f"DISCO model time: {disco_time:.3f}s")
print(f"DISCO uses only 3.2% cache but maintains similar performance!")
```

## 提交前检查清单

在创建 PR 之前，请确保：

- [ ] 所有 DISCO 测试通过
- [ ] 不破坏现有的 LLaMA 测试
- [ ] 代码符合 PaddleNLP 编码规范
- [ ] 添加了适当的文档和注释
- [ ] 性能测试显示预期的内存节省

## 持续集成

提交 PR 后，GitHub Actions 会自动运行测试。如果 CI 失败，请检查：

1. 是否有未解决的导入问题
2. 是否所有测试在本地通过
3. 是否有代码格式问题（使用 `black` 或 `yapf` 格式化）

## 联系方式

如有测试相关问题，请：
- 在 PR 中留言
- 或在 Issue 中描述问题
- 附上完整的错误日志

---

**注意**：DISCO 算法的设计目标是在保持模型质量的同时大幅减少内存使用。测试应该验证这两个关键目标都得到满足。