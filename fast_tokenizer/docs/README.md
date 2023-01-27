# 环境要求

- Linux
- Python >= 3.6

在生成文档前，需要通过以下方式安装 Python 依赖包:

```shell

python -m pip install sphinx sphinx_rtd_theme

```

# 文档生成

FastTokenizer 同时提供 C++ 以及 Python 文档。可在当前目录下执行以下命令同时生成 C++ 以及 Python 文档：

```shell
make html_all
```

## 注意事项

假如文档目录结构发生改变，需要先将之前文档生成的缓存文件清除，再重新生成文档：

```shell
make clean && make html_all
```
