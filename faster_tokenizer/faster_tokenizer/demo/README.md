# FasterTokenizer Demo

## 1. 环境准备

当前版本支持Linux环境：Ubuntu 16.04+，CentOS 7+。
|依赖|版本|
|---|---|
|cmake|>=16.0|
|gcc|>=8.2.0|

1. 下载FasterTokenizer预编译包。

```shell
wget -c https://bj.bcebos.com/paddlenlp/faster_tokenizer/faster_tokenizer_cpp-0.1.0.tar.gz
```

2. 解压。

```shell
tar xvfz faster_tokenizer_cpp-0.1.0.tar.gz
```

## 2. 快速开始

### 2.1 编译

```shell
# 创建编译目录
mkdir build
cd build

# 运行cmake，通过指定faster_tokenizer包的路径，构建Makefile
cmake .. -DFASTER_TOKENIZER_LIB=/path/to/faster_tokenizer_cpp

# 编译
make
```

### 2.2 运行

```shell
./ernie_faster_tokenizer_demo
```
