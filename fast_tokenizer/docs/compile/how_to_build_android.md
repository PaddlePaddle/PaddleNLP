# Android 编译

FastTokenizer 提供两种版本 Android 库，分别是常规版本以及轻量版本。常规版本的 FastTokenizer Android 库功能齐全，可支持任意语言的分词功能，库体积大约为 **32 M**；轻量版本主要支持中文和英文两种语言的分词，库体积约为 **7.4 M**。开发者可以根据自己实际需求选择合适的版本安装，以下将分别介绍这两种版本的编译方式。

## 环境依赖

- cmake >= 3.10
- NDK >= 20

## 配置NDK
```bash
wget https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip
unzip android-ndk-r20b-linux-x86_64.zip # 会解压缩到 android-ndk-r20b 目录
export NDK_ROOT=${PWD}/android-ndk-r20b
```

## 编译 C++ 库方法

### 常规版本

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/fast_tokenizer
mkdir build & cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_NATIVE_API_LEVEL=android-21 -DANDROID_STL=c++_shared -DWITH_TESTING=OFF -DWITH_PYTHON=OFF -DANDROID_TOOLCHAIN=clang
make -j8
```

### 轻量版本

```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/fast_tokenizer
mkdir build & cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_NATIVE_API_LEVEL=android-21 -DANDROID_STL=c++_shared -DWITH_TESTING=OFF -DWITH_PYTHON=OFF -DANDROID_TOOLCHAIN=clang -DWITH_ICU_LITE=ON
make -j8
```

### 库体积压缩

编译后的 C++ 库在当前目录下的 `cpp` 目录下。可以选择使用 strip 减少库体积:
```shell
$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip libcore_tokenizers.so
```

更多编译选项说明参考[编译指南](./README.md)
