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
