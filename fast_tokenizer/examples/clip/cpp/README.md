# ClipFastTokenizer C++ 示例

## 1. 快速安装

当前版本FastTokenizer C++库支持不同的操作系统以及硬件平台，用户可以根据实际的使用环境，从以下选择合适的预编译包：
|系统|下载地址|
|---|---|
|Linux-x64| [fast_tokenizer-linux-x64-1.0.0.tgz](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-x64-1.0.0.tgz) |
|Linux-aarch64| [fast_tokenizer-linux-aarch64-1.0.0.tgz](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-aarch64-1.0.0.tgz) |
|Windows| [fast_tokenizer-win-x64-1.0.0.zip](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-win-x64-1.0.0.zip) |
|MacOS-x64| [fast_tokenizer-osx-x86_64-1.0.0.tgz](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-osx-x86_64-1.0.0.tgz) |
|MacOS-arm64| [fast_tokenizer-osx-arm64-1.0.0.tgz](https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-osx-arm64-1.0.0.tgz) |

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

以下以Linux平台为例, 介绍如何使用FastTokenizer C++预编译包完成demo示例编译及运行。该示例会生成一个名为`demo`的可执行文件。

### 2.1 下载解压

```shell
wget -c https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-x64-1.0.0.tgz

tar xvfz fast_tokenizer-linux-x64-1.0.0.tgz
# 解压后为fast_tokenizer目录
```

解压后得到fast_tokenizer目录，该目录的结构如下：

```shell

fast_tokenizer
|__ commit.log              # 编译时的commit id
|__ FastTokenizer.cmake     # FastTokenizer CMake文件，定义了头文件目录、动态链接库目录变量
|__ include                 # FastTokenizer的头文件目录
|__ lib                     # FastTokenizer的动态链接库目录
|__ third_party             # FastTokenizer依赖的第三方库目录

```

推荐用户直接使用cmake方式引入FastTokenizer库。在CMake引入FastTokenizer时，只需添加一行 `include(FastTokenizer.cmake)`，即可获取FastTokenizer的预定义的CMake变量`FAST_TOKENIZER_INCS`和`FAST_TOKENIZER_LIBS`，分别指定FastTokenizer的头文件目录以及动态链接库目录。


### 2.2 编译

示例提供简单的CMakeLists.txt, 用户仅需指定fast_tokenizer包的路径，即可完成编译。

```shell

# 创建编译目录
mkdir build
cd build

# 运行cmake，通过指定fast_tokenizer包的路径，构建Makefile
cmake .. -DFAST_TOKENIZER_INSTALL_DIR=/path/to/fast_tokenizer

# 编译
make

```

### 2.3 运行

```shell
./demo
```


### 2.4 样例输出

输出包含原始文本的输入，以及分词后的ids序列结果（含padding）。

```shell

text = "a photo of an astronaut riding a horse on mars"
ids = [49406, 320, 1125, 539, 550, 18376, 6765, 320, 4558, 525, 7496, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407]

```
