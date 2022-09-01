import unittest
from typing import Optional, Union
from paddlenlp.transformers.configuration_utils import parse_config, PretrainedConfig, parse_config_and_model
from paddlenlp.transformers.model_utils import PretrainedModel


class FakePretrainedModelConfig(PretrainedConfig):

    def __init__(self, a=0, b=1, c=2):
        self.a = a
        self.b = b
        self.c = c


class FakeLayer:

    def __init__(self,
                 config: Optional[FakePretrainedModelConfig] = None,
                 *args,
                 **kwargs):
        super(FakeLayer, self).__init__()

        config: FakePretrainedModelConfig = parse_config(
            config=config,
            config_class=FakePretrainedModelConfig,
            args=args,
            kwargs=kwargs,
            fields=["a", "b", "c"],
        )

        self.a = config.a
        self.b = config.b
        self.c = config.c


class FakeModel(PretrainedModel):

    def __init__(
            self,
            config_or_model: Optional[Union[FakeLayer,
                                            FakePretrainedModelConfig]] = None,
            *args,
            **kwargs):
        """fake `__init__`, the source of parameters is:

            def __init__(self, model, a, b):
                self.model = model
                self.a = a
                self.b = b

        Args:
            config_or_model (Optional[Union[FakeLayer, FakePretrainedModelConfig]], optional): config or model instance. Defaults to None.
        """
        super().__init__()

        config, model = parse_config_and_model(
            config_or_model=config_or_model,
            config_class=FakePretrainedModelConfig,
            model_class=FakeLayer,
            args=args,
            kwargs=kwargs,
            fields=["a", "b"],
        )

        self.model: FakeLayer = model
        self.a = config.a
        self.b = config.b


class ConfigurationUtilsTest(unittest.TestCase):

    def test_parse_config_with_single_config(self):
        # 1. single config
        config = FakePretrainedModelConfig(a=10, b=11, c=12)
        model = FakeLayer(config)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_with_full_kwargs(self):
        model = FakeLayer(a=10, b=11, c=12)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_with_full_args(self):
        model = FakeLayer(10, 11, 12)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_with_args_and_kwargs(self):
        model = FakeLayer(10, b=11, c=12)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_and_model_with_single_config(self):
        config = FakePretrainedModelConfig(a=10, b=11, c=12)
        model = FakeModel(config)
        assert model.a == 10
        assert model.b == 11

    def test_parse_config_and_model_with_model_and_args(self):
        config = FakePretrainedModelConfig(a=10, b=11, c=12)
        fake_layer = FakeLayer(config)
        model = FakeModel(fake_layer, 100, 110)
        assert model.a == 100
        assert model.b == 110

        assert model.model.a == 10
        assert model.model.b == 11

    def test_parse_config_and_model_with_model_and_kwargs(self):
        config = FakePretrainedModelConfig(a=10, b=11, c=12)
        fake_layer = FakeLayer(config)
        model = FakeModel(fake_layer, a=100, b=110)

        assert model.a == 100
        assert model.b == 110

        assert model.model.a == 10
        assert model.model.b == 11

    def test_parse_config_and_model_with_model_and_kwargs_and_args(self):
        config = FakePretrainedModelConfig(a=10, b=11, c=12)
        fake_layer = FakeLayer(config)
        model = FakeModel(fake_layer, 100, b=110)

        assert model.a == 100
        assert model.b == 110

        assert model.model.a == 10
        assert model.model.b == 11
