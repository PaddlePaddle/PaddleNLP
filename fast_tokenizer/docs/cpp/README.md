# FastTokenizer C++ 库使用教程

## 1. 快速安装

当前版本FastTokenizer C++库支持不同的操作系统以及硬件平台，并为以下平台提供预编译包：
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

### 下载解压

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


## 2. 快速开始

目前FastTokenizer提供了以下C++使用示例。

[ErnieFastTokenizer C++示例](../../examples/ernie/)

[ClipFastTokenizer C++示例](../../examples/clip/)
