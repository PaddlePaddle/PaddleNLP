# Android 编译

## 环境依赖

- cmake >= 3.10
- NDK >= 20

## 配置NDK
```bash
wget https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip
unzip android-ndk-r20b-linux-x86_64.zip # 会解压缩到 android-ndk-r20b 目录
export NDK_ROOT=${PWD}/android-ndk-r20b
```

## 编译C++库方法

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/fast_tokenizer
mkdir build & cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_NATIVE_API_LEVEL=android-21 -DANDROID_STL=c++_static -DWITH_TESTING=OFF -DWITH_PYTHON=OFF -DANDROID_TOOLCHAIN=clang
make -j8
```
编译后的C++库在当前目录下的`cpp`目录下。可以选择使用strip减少库体积:
```shell
$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip libcore_tokenizers.so
```

更多编译选项说明参考[编译指南](./README.md)
