# cpu-custom-ops

## 快速开始
### 1.环境准备
```shell
# 查询机器是否支持 avx512指令
lscpu | grep avx512*
```

### 2.安装 cpu 自定义算子和第三方库
```shell
#建议在 gcc 9.4.0 下安装第三方库
bash setup.sh
```
**Note:**

包含 avx 指令 CPU 机器大模型推理教程 [X86 CPU](../../llm/docs/cpu_install.md)
