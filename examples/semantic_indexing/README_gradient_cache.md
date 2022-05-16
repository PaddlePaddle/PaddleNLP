### “数据准备”参考facebook仓库download_data.py下载即可，下载后放到对应文件夹里。
### ”模型训练“运行train_gradient_cache_DPR.py即可，需按照原始论文仓库中的最佳训练策略设置参数，并设置模型保存位置。
### ”效果评估“先运行generate_dense_embeddings.py文件，之后运行dense_retriever.py文件即可。
### 效果评估相关文件从facebook的库中取得，将其中用torch实现的过程改为了用paddle实现
##上述文件的运行参数与DPR原库一致
##train里面把global的量改成参数就行
