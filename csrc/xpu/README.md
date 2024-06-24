# ernie-bot-custom-ops
ernie bot 昆仑自定义算子库。

## 快速开始
# 构建 XDNN plugin 和 Paddle 自定义算子库
```
$ cd src
$ wget https://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev/20240429/xdnn-ubuntu_x86_64.tar.gz
$ wget https://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev/20240429/xre-ubuntu_x86_64.tar.gz
$ wget -q --no-check-certificate https://klx-sdk-release-public.su.bcebos.com/xtdk_llvm15/dev/2.7.98.2/xtdk-llvm15-ubuntu1604_x86_64.tar.gz
$ tar -xf xdnn-ubuntu_x86_64.tar.gz
$ tar -xf xre-ubuntu_x86_64.tar.gz
$ tar -xf xtdk-llvm15-ubuntu1604_x86_64.tar.gz
$ export PWD=$(pwd)
$ export XDNN_PATH=${PWD}/xdnn-ubuntu_x86_64/
$ export XRE_PATH=${PWD}/xre-ubuntu_x86_64/
$ export CLANG_PATH=${PWD}/xtdk-llvm15-ubuntu1604_x86_64/
$ bash ./cmake_build.sh
```

## 测试
# 运行 add2 单测
```
$ cd test/python
$ python test_get_padding_offset_v2.py
```

## 如何贡献
```
$ pip install pre-commit==2.17.0
$ pre-commit install
```
