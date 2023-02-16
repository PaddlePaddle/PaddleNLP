# FastTokenizer 编译指南

本文档说明编译 FastTokenizer C++ 库、Python 库以及 Android 库三种编译过程，根据编译的平台参考如下文档

- [Linux & Mac 编译](./how_to_build_linux_and_mac.md)
- [Windows 编译](./how_to_build_windows.md)
- [Android 编译](./how_to_build_android.md)

FastTokenizer 使用 CMake 编译，其中编译过程中，各平台上编译选项如下表所示

| 选项 | 作用 | 备注 |
|:---- | :--- | :--- |
| WITH_PYTHON | 是否编译为 Python 库，默认为是，否则为 C++ 库||
| WITH_TESTING | 是否编译 C++ 单测，默认为否 ||
| WITH_ICU_LITE | 是否与 ICU Lite 依赖包联编，打开后可减小 FastTokenizer 库体积，默认为否 | 只能用于 Andriod 编译，打开后 FastTokenizer 库体积大小从 **32 M 减少到 7.4 M**，但只能对中英文进行分词。|
| USE_ABI0 | 是否编译_GLIBCXX_USE_CXX11_ABI=0, 默认为OFF。|
