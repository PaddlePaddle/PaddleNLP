# Linux & Mac编译

## 环境依赖

- cmake >= 3.10
- gcc >= 8.2.0

## 编译 C++ 库方法

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/fast_tokenizer
mkdir build & cd build
cmake .. -DWITH_PYTHON=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
make -j8
```

编译后的 C++ 库在当前目录下的 `cpp` 目录下。

## 编译 Python 库方法

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/fast_tokenizer
mkdir build & cd build
# 设置 Python 环境
export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}

cmake .. -DWITH_PYTHON=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
make -j8
```

编译后的 wheel 包即在当前目录下的 `dist` 目录中

更多编译选项说明参考[编译指南](./README.md)
