
# LLM CE 单测

## 执行方式

```shell
cd ./tests
export PYTHONPATH=../:$PYTHONPATH
MODEL_NAME=llama pytest -v -s ./test_tipc/llm/test_predictor.py
```

> 脚本为了和 CE 测试脚本路径一致，故此处需要以./tests 为执行目录。

`MODEL_NAME` 的取值来源于 ./tests/test_tipc/llm/fixtures/predictor.yaml 文件中的一级 key name。

在单测的执行日志文件中，会打印：`IPS: xxx`等数据用来标识对应 benchmark 数据，在报表中只需要取最后一个数据即可：推理模型静态图的推理性能数据。
