import os
import tempfile
import unittest

import pytest
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoModel, CLIPTextModel, CLIPModel


class ModelLoadTester(unittest.TestCase):
    @pytest.mark.skip
    def test_config_diff(self, config_1, config_2):
        config_1 = config_1.to_dict()
        config_2 = config_2.to_dict()
        config_1.pop("architectures", None)
        config_2.pop("architectures", None)
        assert config_1 == config_2, "config not equal"

    
    def test_clip_load(self):
        # BOS
        logger.info("Download model from PaddleNLP BOS")
        # 从bos下载非use_safetensors的模型文件
        clip_model_bos = CLIPTextModel.from_pretrained("baicai/tiny-clip", use_safetensors=False, from_hf_hub=False)
        # 测试从cache加载模型文件
        clip_model_bos_auto = AutoModel.from_pretrained("baicai/tiny-clip", use_safetensors=False, from_hf_hub=False)
        self.test_config_diff(clip_model_bos.config, clip_model_bos_auto.config)

        logger.info("Download model from PaddleNLP BOS with subfolder")
        # 测试bos存在subfolder时下载情况
        clip_model_bos_sub = CLIPTextModel.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="tiny-clip", use_safetensors=False, from_hf_hub=False
        )
        self.test_config_diff(clip_model_bos.config, clip_model_bos_sub.config)

        # 测试从cache加载模型且存在subfolder
        clip_model_bos_sub_auto = AutoModel.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="tiny-clip", use_safetensors=False, from_hf_hub=False
        )
        self.test_config_diff(clip_model_bos_sub.config, clip_model_bos_sub_auto.config)



        # aistudio
        logger.info("Download model from aistudio")
        # 从aistudio下载非use_safetensors的模型文件
        clip_model_aistudio = CLIPTextModel.from_pretrained(
            "aistudio/tiny-clip", use_safetensors=False, from_aistudio=True
        )
        self.test_config_diff(clip_model_bos.config, clip_model_aistudio.config)

        # 测试从cache加载模型文件
        clip_model_aistudio_auto = AutoModel.from_pretrained(
            "aistudio/tiny-clip", use_safetensors=False, from_aistudio=True
        )
        self.test_config_diff(clip_model_aistudio.config, clip_model_aistudio_auto.config)

        logger.info("Download model from aistudio with subfolder")
        # 测试aistudio存在subfolder时下载情况
        clip_model_aistudio_sub = CLIPTextModel.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="tiny-clip", use_safetensors=False, from_aistudio=True
        )
        self.test_config_diff(clip_model_aistudio.config, clip_model_aistudio_sub.config)

        # 测试从cache加载模型且存在subfolder
        clip_model_aistudio_sub_auto = AutoModel.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="tiny-clip", use_safetensors=False, from_aistudio=True
        )
        self.test_config_diff(clip_model_aistudio_sub.config, clip_model_aistudio_sub_auto.config)



        # hf
        logger.info("Download model from hf")
        # 从hf下载非use_safetensors的模型文件
        clip_model_hf = CLIPTextModel.from_pretrained(
            "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=False
        )
        self.test_config_diff(clip_model_hf.config, clip_model_hf.config)

        # 测试从cache加载模型文件
        clip_model_hf_auto = AutoModel.from_pretrained(
            "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=False
        )
        self.test_config_diff(clip_model_hf.config, clip_model_hf_auto.config)

        logger.info("Download model from hf with subfolder")
        # 测试hf存在subfolder时下载情况
        clip_model_hf_sub = CLIPTextModel.from_pretrained(
            "Baicai003/paddlenlp-test-model", subfolder="tiny-clip-one", from_hf_hub=True, use_safetensors=False
        )
        self.test_config_diff(clip_model_hf.config, clip_model_hf_sub.config)
        # 测试从cache加载模型且存在subfolder
        clip_model_hf_sub_auto = AutoModel.from_pretrained(
            "Baicai003/paddlenlp-test-model", subfolder="tiny-clip-one", from_hf_hub=True, use_safetensors=False
        )
        self.test_config_diff(clip_model_hf_sub.config, clip_model_hf_sub_auto.config)



        # modelscope
        logger.info("Download model from modelscope")
        os.environ['from_modelscope'] = 'True'

        # 从modelscope下载非use_safetensors的模型文件
        clip_auto_model_scope = AutoModel.from_pretrained('xiaoguailin/clip-vit-large-patch14', use_safetensors=False)

        # 测试从cache加载模型文件
        clip_model_scope = CLIPModel.from_pretrained('xiaoguailin/clip-vit-large-patch14', use_safetensors=False, convert_from_torch=True)
        self.test_config_diff(clip_auto_model_scope.config, clip_model_scope.config)

        # logger.info("Download model from hf with subfolder")
        # # 测试modelscope存在subfolder时下载情况
        # clip_model_scope = CLIPModel.from_pretrained("xiaoguailin", subfolder="clip-vit-large-patch14", use_safetensors=False, convert_from_torch=True)
        # self.test_config_diff(clip_auto_model_scope.config, clip_model_scope.config)
    
        # # 测试从cache加载模型且存在subfolder
        # clip_model_scope = CLIPModel.from_pretrained("xiaoguailin", subfolder="clip-vit-large-patch14", use_safetensors=False, convert_from_torch=True)
        # self.test_config_diff(clip_auto_model_scope.config, clip_model_scope.config)
        # os.environ['from_modelscope'] = 'False'



        # local
        logger.info("Download model from local")
        # 将文件保存到本地
        clip_model_bos.save_pretrained("./paddlenlp-test-model/tiny-clip", safe_serialization=False)
        # 测试本地文件加载
        clip_model_local = AutoModel.from_pretrained("./paddlenlp-test-model/tiny-clip", use_safetensors=False)
        self.test_config_diff(clip_model_bos.config, clip_model_local.config)
        # 测试本地存在subfolder时文件加载
        clip_model_local_subfolder = AutoModel.from_pretrained("./paddlenlp-test-model/", subfolder="tiny-clip", use_safetensors=False)
        self.test_config_diff(clip_model_local.config, clip_model_local_subfolder.config)



        # 从build-in中获取url，直接从url进行下载
        logger.info('url')
        AutoModel.from_pretrained('t5-small', from_hf_hub=True, use_safetensors=False)
        AutoModel.from_pretrained('t5-small', from_aistudio=True, use_safetensors=False)


    def test_clip_load_safe(self):
        # BOS
        logger.info("Download model from PaddleNLP BOS")
        # 从bos下载use_safetensors的模型文件
        clip_model_bos = CLIPTextModel.from_pretrained("baicai/tiny-clip", use_safetensors=True, from_hf_hub=False)
        # 测试从cache加载模型文件
        clip_model_bos_auto = AutoModel.from_pretrained("baicai/tiny-clip", use_safetensors=True, from_hf_hub=False)
        self.test_config_diff(clip_model_bos.config, clip_model_bos_auto.config)

        logger.info("Download model from PaddleNLP BOS with subfolder")
        # 测试bos存在subfolder时下载情况
        clip_model_bos_sub = CLIPTextModel.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="tiny-clip", use_safetensors=True, from_hf_hub=False
        )
        self.test_config_diff(clip_model_bos.config, clip_model_bos_sub.config)

        # 测试从cache加载模型且存在subfolder
        clip_model_bos_sub_auto = AutoModel.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="tiny-clip", use_safetensors=True, from_hf_hub=False
        )
        self.test_config_diff(clip_model_bos_sub.config, clip_model_bos_sub_auto.config)



        # aistudio
        logger.info("Download model from aistudio")
        # 从aistudio下载use_safetensors的模型文件
        clip_model_aistudio = CLIPTextModel.from_pretrained(
            "aistudio/tiny-clip", use_safetensors=True, from_aistudio=True
        )
        self.test_config_diff(clip_model_bos.config, clip_model_aistudio.config)
        # 测试从cache加载模型文件
        clip_model_aistudio_auto = AutoModel.from_pretrained(
            "aistudio/tiny-clip", use_safetensors=True, from_aistudio=True
        )
        self.test_config_diff(clip_model_aistudio.config, clip_model_aistudio_auto.config)

        logger.info("Download model from aistudio with subfolder")
        # 测试aistudio存在subfolder时下载情况
        clip_model_aistudio_sub = CLIPTextModel.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="tiny-clip", use_safetensors=True, from_aistudio=True
        )
        self.test_config_diff(clip_model_aistudio.config, clip_model_aistudio_sub.config)
        # 测试从cache加载模型且存在subfolder
        clip_model_aistudio_sub_auto = AutoModel.from_pretrained(
            "aistudio/paddlenlp-test-model", subfolder="tiny-clip", use_safetensors=True, from_aistudio=True
        )
        self.test_config_diff(clip_model_aistudio_sub.config, clip_model_aistudio_sub_auto.config)



        # hf
        logger.info("Download model from hf")
        # 从hf下载use_safetensors的模型文件
        clip_model_hf = CLIPTextModel.from_pretrained(
            "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=True
        )
        self.test_config_diff(clip_model_hf.config, clip_model_hf.config)
        # 测试从cache加载模型文件
        clip_model_hf_auto = AutoModel.from_pretrained(
            "Baicai003/tiny-clip-one", from_hf_hub=True, use_safetensors=True
        )
        self.test_config_diff(clip_model_hf.config, clip_model_hf_auto.config)

        logger.info("Download model from hf with subfolder")
        # 测试hf存在subfolder时下载情况
        clip_model_hf_sub = CLIPTextModel.from_pretrained(
            "Baicai003/paddlenlp-test-model", subfolder="tiny-clip-one", from_hf_hub=True, use_safetensors=True
        )
        self.test_config_diff(clip_model_hf.config, clip_model_hf_sub.config)
        # 测试从cache加载模型且存在subfolder
        clip_model_hf_sub_auto = AutoModel.from_pretrained(
            "Baicai003/paddlenlp-test-model", subfolder="tiny-clip-one", from_hf_hub=True, use_safetensors=True
        )
        self.test_config_diff(clip_model_hf_sub.config, clip_model_hf_sub_auto.config)



        # modelscope
        logger.info("Download model from modelscope")
        os.environ['from_modelscope'] = 'True'

        # 从modelscope下载use_safetensors的模型文件
        clip_auto_model_scope = AutoModel.from_pretrained('xiaoguailin/clip-vit-large-patch14', use_safetensors=True)

        # 测试从cache加载模型文件
        clip_model_scope = CLIPModel.from_pretrained('xiaoguailin/clip-vit-large-patch14', use_safetensors=True)
        self.test_config_diff(clip_auto_model_scope.config, clip_model_scope.config)

        # logger.info("Download model from hf with subfolder")
        # # 测试modelscope存在subfolder时下载情况
        # clip_model_scope = CLIPModel.from_pretrained("xiaoguailin", subfolder="clip-vit-large-patch14", use_safetensors=True)
        # self.test_config_diff(clip_auto_model_scope.config, clip_model_scope.config)
    
        # # 测试从cache加载模型且存在subfolder
        # clip_model_scope = CLIPModel.from_pretrained("xiaoguailin", subfolder="clip-vit-large-patch14", use_safetensors=True)
        # self.test_config_diff(clip_auto_model_scope.config, clip_model_scope.config)
        # os.environ['from_modelscope'] = 'False'



        # local
        logger.info("Download model from local")
        # 将文件保存到本地
        clip_model_bos.save_pretrained("./paddlenlp-test-model/tiny-clip", safe_serialization=True)
        # 测试本地文件加载
        clip_model_local = CLIPTextModel.from_pretrained("./paddlenlp-test-model/tiny-clip", use_safetensors=True)
        self.test_config_diff(clip_model_bos.config, clip_model_local.config)
        clip_model_local_auto = AutoModel.from_pretrained("./paddlenlp-test-model/", subfolder="tiny-clip", use_safetensors=True)
        self.test_config_diff(clip_model_local.config, clip_model_local_auto.config)
        


        # 从build-in中获取url，直接从url进行下载
        logger.info('url')
        AutoModel.from_pretrained('t5-small', from_hf_hub=True)
        AutoModel.from_pretrained('t5-small', from_aistudio=True)


test = ModelLoadTester()
test.test_clip_load()
test.test_clip_load_safe()