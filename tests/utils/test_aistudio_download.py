import unittest
from paddlenlp.utils.download import resolve_file_path
from paddlenlp.transformers.aistudio_utils import aistudio_download


class TestAistudioDownload(unittest.TestCase):

    def test_aistudio_download(self):
        # 设置测试数据
        repo_id = 'PaddleNLP/DeepSeek-R1-Distill-Qwen-1.5B'
        filename = 'model.safetensors'
        revision = 'master'
        local_dir = './local/model'

        # 调用待测试的函数
        result = resolve_file_path(
            repo_id=repo_id,
            filenames=filename,
            revision=revision,
            from_aistudio=True,
            local_dir=local_dir,
        )

        # 验证结果
        print(result)

    def test_aistudio_download_transformer(self):
        # 设置测试数据
        repo_id = 'PaddleNLP/DeepSeek-R1-Distill-Qwen-1.5B'
        filename = 'model.safetensors'
        revision = 'master'
        cache_dir = './local/model'

        # 调用待测试的函数
        result = aistudio_download(
            repo_id=repo_id,    
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )

        # 验证结果
        print(result)

if __name__ == '__main__':
    unittest.main()