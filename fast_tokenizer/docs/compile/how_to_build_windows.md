# Windows 编译

## 环境依赖

- cmake >= 3.10
- VS 2019
- ninja
- cmake >= 3.10

以上依赖安装好后，在 Windows 菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具即可进行下面的编译环节。

## 编译 C++ 库方法

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/fast_tokenizer
mkdir build & cd build
cmake .. -G "Ninja" -DWITH_PYTHON=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
ninja -j8
```

编译后的 C++ 库在当前目录下的`cpp`目录下。

## 编译 Python 库方法

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/fast_tokenizer
mkdir build & cd build
# 需要指定 Python 库
cmake .. -G "Ninja" -DWITH_PYTHON=ON ^
                    -DWITH_TESTING=OFF ^
                    -DCMAKE_BUILD_TYPE=Release ^
                    -DPYTHON_EXECUTABLE=C:\Python37\python.exe ^
                    -DPYTHON_INCLUDE_DIR=C:\Python37\include ^
                    -DPYTHON_LIBRARY=C:\Python37\libs\python3%%x.lib
ninja -j8
```

编译后的 wheel 包即在当前目录下的 `dist` 目录中

更多编译选项说明参考[编译指南](./README.md)
