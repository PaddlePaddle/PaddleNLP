# 指定语义检索系统的Yaml配置文件
export CUDA_VISIBLE_DEVICES=0
export PIPELINE_YAML_PATH=rest_api/pipeline/semantic_search.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891