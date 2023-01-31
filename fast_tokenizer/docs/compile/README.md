# FastTokenizer 编译指南

本文档说明编译 FastTokenizer C++ 库、Python 库两种编译过程，根据编译的平台参考如下文档

- [Linux & Mac 编译](./how_to_build_linux_and_mac.md)
- [Windows 编译](./how_to_build_windows.md)

FastTokenizer 使用 CMake 编译，其中编译过程中，各平台上编译选项如下表所示

| 选项 | 作用 | 备注 |
|:---- | :--- | :--- |
| WITH_PYTHON | 是否编译为 Python 库，默认为是，否则为 C++ 库|
| WITH_TESTING | 是否编译 C++ 单测，默认为否 |
