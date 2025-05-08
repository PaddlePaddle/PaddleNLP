# WINTX Triton kernel

## 功能介绍
- 支持 wint2.75/wint2.5/wint4 moe 推理
- 支持 wint4 gemm 推理

## 安装依赖

```bash
python -m pip install triton==3.0
python -m pip install git+https://github.com/zhoutianzi666/UseTritonInPaddle.git
python -c "import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()"
```

> wint2.75/wint2.5 moe kernel 是共用同一 kernel，并使用静态导入的方式，第一次推理会编译，后续推理会直接使用编译好的 kernel。
> 默认编译好的 kernel 会存放在`~/.paddlenlp/triton_ops_rank_*`目录下
