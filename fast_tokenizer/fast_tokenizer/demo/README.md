# FastTokenizer C++ Demo

## 1. 快速安装

当前版本FastTokenizer C++库支持不同的操作系统以及硬件平台，并为以下平台提供预编译包：
|系统|下载地址|
|---|---|
|Linux-x64| [fast_tokenizer-linux-x64-dev.tgz](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-x64-dev.tgz) |
|Linux-aarch64| [fast_tokenizer-linux-aarch64-dev.tgz](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-aarch64-dev.tgz) |
|Windows| [fast_tokenizer-win-x64-dev.zip](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-win-x64-dev.zip) |
|MacOS-x64| [fast_tokenizer-osx-x86_64-dev.tgz](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-osx-x86_64-dev.tgz) |
|MacOS-arm64| [fast_tokenizer-osx-arm64-dev.tgz](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-osx-arm64-dev.tgz) |

### 环境依赖

#### 系统环境要求
|系统|版本|
|---|---|
|Linux|Ubuntu 16.04+，CentOS 7+|
|Windows|10|
|MacOS| 11.4+|


#### Linux，Mac编译环境要求
|依赖|版本|
|---|---|
|cmake|>=16.0|
|gcc|>=8.2.0|

#### Windows编译环境要求
|依赖|版本|
|---|---|
|cmake|>=16.0|
|VisualStudio|2019|

## 2. 快速开始

以下以Linux平台为例, 介绍如何使用FastTokenizer CPP预编译包完成demo示例编译及运行。

### 2.1 下载解压

```shell
wget -c https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-x64-dev.tgz

tar xvfz fast_tokenizer-linux-x64-dev.tgz
# 解压后为fast_tokenizer目录
```

### 2.1 编译

```shell
# 创建编译目录
mkdir build
cd build

# 运行cmake，通过指定fast_tokenizer包的路径，构建Makefile
cmake .. -DFAST_TOKENIZER_INSTALL_DIR=/path/to/fast_tokenizer

# 编译
make
```

### 2.2 运行

```shell
./ernie_fast_tokenizer_demo
```


### 2.3 样例输出

```shell
case 1: Tokenize a single string
The Encoding content:
ids: 1, 278, 1612, 375, 497, 837, 793, 9, 577, 53, 230, 129, 37, 998, 195, 381, 441, 28, 233, 406, 1572, 276, 505, 110, 51, 53, 230, 2718, 17, 17, 130, 337, 612, 5, 104, 49, 598, 592, 358, 1137, 1890, 5, 53, 612, 2
type_ids: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
tokens: [CLS], 商, 赢, 环, 球, 股, 份, 有, 限, 公, 司, 关, 于, 延, 期, 回, 复, 上, 海, 证, 券, 交, 易, 所, 对, 公, 司, 2017, 年, 年, 度, 报, 告, 的, 事, 后, 审, 核, 问, 询, 函, 的, 公, 告, [SEP]
offsets: (0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (0, 0)
special_tokens_mask: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
attention_mask: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
sequence_ranges: {0 : (1, 44) },
```
