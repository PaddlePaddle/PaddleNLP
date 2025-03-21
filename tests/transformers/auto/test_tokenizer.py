# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Hugging Face inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import unittest

import pytest

import paddlenlp
from paddlenlp.transformers import (
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
    BertTokenizerFast,
    CTRLTokenizer,
    GPTTokenizer,
    GPTTokenizerFast,
    PretrainedTokenizerFast,
    RobertaConfig,
    RobertaTokenizer,
    RobertaTokenizerFast,
)
from paddlenlp.transformers.auto.configuration import CONFIG_MAPPING, AutoConfig
from paddlenlp.transformers.auto.tokenizer import (
    TOKENIZER_MAPPING,
    get_tokenizer_config,
    tokenizer_class_from_name,
)
from paddlenlp.utils.env import TOKENIZER_CONFIG_NAME
from paddlenlp.utils.import_utils import is_tokenizers_available

from ...testing_utils import (
    DUMMY_DIFF_TOKENIZER_IDENTIFIER,
    DUMMY_UNKNOWN_IDENTIFIER,
    SMALL_MODEL_IDENTIFIER,
    RequestCounter,
    require_tokenizers,
    slow,
)

# Used to test Auto{Config, Model, Tokenizer} model_type detection.
from ...utils.test_module.custom_configuration import CustomConfig
from ...utils.test_module.custom_tokenizer import CustomTokenizer

if is_tokenizers_available():
    from ...utils.test_module.custom_tokenizer_fast import (
        CustomTokenizerFast,
        CustomTokenizerFastWithoutSlow,
    )


class AutoTokenizerTest(unittest.TestCase):
    @unittest.skip("skipping due to connection error!")
    def test_from_aistudio(self):
        tokenizer = AutoTokenizer.from_pretrained("PaddleNLP/tiny-random-bert", from_aistudio=True)
        self.assertIsInstance(tokenizer, paddlenlp.transformers.BertTokenizer)

    def test_from_pretrained_cache_dir(self):
        model_name = "__internal_testing__/tiny-random-bert"
        with tempfile.TemporaryDirectory() as tempdir:
            AutoTokenizer.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, TOKENIZER_CONFIG_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))

    def test_from_pretrained_tokenizer_fast(self):
        tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2", use_fast=True)
        self.assertIsInstance(tokenizer, BertTokenizerFast)

    @slow
    def test_tokenizer_from_pretrained(self):
        for model_name in {"google-bert/bert-base-uncased", "google-bert/bert-base-cased"}:
            tokenizer = AutoTokenizer.from_pretrained(model_name, from_hf_hub=True)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
            self.assertGreater(len(tokenizer), 0)

        for model_name in ["openai-community/gpt2", "openai-community/gpt2-medium"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name, from_hf_hub=True)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, (GPTTokenizer, GPTTokenizerFast))
            self.assertGreater(len(tokenizer), 0)

    def test_tokenizer_from_pretrained_identifier(self):
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER, from_hf_hub=True)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 12)

    def test_tokenizer_from_model_type(self):
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, from_hf_hub=True)
        self.assertIsInstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 20)

    def test_tokenizer_from_tokenizer_class(self):
        config = AutoConfig.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER, from_hf_hub=True)
        self.assertIsInstance(config, RobertaConfig)
        # Check that tokenizer_type ≠ model_type
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER, config=config, from_hf_hub=True)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 12)

    def test_tokenizer_from_type(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy("./tests/fixtures/vocab.txt", os.path.join(tmp_dir, "vocab.txt"))

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type="bert", use_fast=False)
            self.assertIsInstance(tokenizer, BertTokenizer)

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy("./tests/fixtures/vocab.json", os.path.join(tmp_dir, "vocab.json"))
            shutil.copy("./tests/fixtures/merges.txt", os.path.join(tmp_dir, "merges.txt"))

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type="gpt2", use_fast=False)
            self.assertIsInstance(tokenizer, GPTTokenizer)

    @require_tokenizers
    def test_tokenizer_from_type_fast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy("./tests/fixtures/vocab.txt", os.path.join(tmp_dir, "vocab.txt"))

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type="bert")
            self.assertIsInstance(tokenizer, BertTokenizerFast)

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy("./tests/fixtures/vocab.json", os.path.join(tmp_dir, "vocab.json"))
            shutil.copy("./tests/fixtures/merges.txt", os.path.join(tmp_dir, "merges.txt"))

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type="gpt2")
            self.assertIsInstance(tokenizer, GPTTokenizerFast)

    def test_tokenizer_from_type_incorrect_name(self):
        with pytest.raises(ValueError):
            AutoTokenizer.from_pretrained("./", tokenizer_type="xxx")

    @require_tokenizers
    def test_tokenizer_identifier_with_correct_config(self):
        for tokenizer_class in [BertTokenizer, BertTokenizerFast, AutoTokenizer]:
            tokenizer = tokenizer_class.from_pretrained("wietsedv/bert-base-dutch-cased", from_hf_hub=True)
            self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))

            if isinstance(tokenizer, BertTokenizer):
                self.assertEqual(tokenizer.basic_tokenizer.do_lower_case, False)
            else:
                self.assertEqual(tokenizer.do_lower_case, False)

            self.assertEqual(tokenizer.model_max_length, 512)

    @require_tokenizers
    def test_tokenizer_identifier_non_existent(self):
        for tokenizer_class in [BertTokenizer, BertTokenizerFast, AutoTokenizer]:
            with self.assertRaisesRegex(
                EnvironmentError,
                "julien-c/herlolip-not-exists is not a local folder and is not a valid model identifier",
            ):
                _ = tokenizer_class.from_pretrained("julien-c/herlolip-not-exists", from_hf_hub=True)

    def test_model_name_edge_cases_in_mappings(self):
        # tests: https://github.com/huggingface/transformers/pull/13251
        # 1. models with `-`, e.g. xlm-roberta -> xlm_roberta
        # 2. models that don't remap 1-1 from model-name to model file, e.g., openai-gpt -> openai
        tokenizers = TOKENIZER_MAPPING.values()
        tokenizer_names = []

        for slow_tok, fast_tok in tokenizers:
            if slow_tok is not None:
                tokenizer_names.append(slow_tok.__name__)

            if fast_tok is not None:
                tokenizer_names.append(fast_tok.__name__)

        for tokenizer_name in tokenizer_names:
            # must find the right class
            tokenizer_class_from_name(tokenizer_name)

    @require_tokenizers
    def test_from_pretrained_use_fast_toggle(self):
        self.assertIsInstance(
            AutoTokenizer.from_pretrained("google-bert/bert-base-cased", use_fast=False, from_hf_hub=True),
            BertTokenizer,
        )
        self.assertIsInstance(
            AutoTokenizer.from_pretrained("google-bert/bert-base-cased", from_hf_hub=True), BertTokenizerFast
        )

    @require_tokenizers
    def test_do_lower_case(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased", do_lower_case=False, from_hf_hub=True
        )
        sample = "Hello, world. How are you?"
        tokens = tokenizer.tokenize(sample)
        self.assertEqual("[UNK]", tokens[0])

        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base", do_lower_case=False, from_hf_hub=True)
        tokens = tokenizer.tokenize(sample)
        self.assertEqual("[UNK]", tokens[0])

    @require_tokenizers
    def test_PretrainedTokenizerFast_from_pretrained(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "robot-test/dummy-tokenizer-fast-with-model-config", from_hf_hub=True
        )
        self.assertEqual(type(tokenizer), PretrainedTokenizerFast)
        self.assertEqual(tokenizer.model_max_length, 512)
        self.assertEqual(tokenizer.vocab_size, 30000)
        self.assertEqual(tokenizer.unk_token, "[UNK]")
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(tokenizer.truncation_side, "right")

    def test_auto_tokenizer_from_local_folder(self):
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER, from_hf_hub=True)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            tokenizer2 = AutoTokenizer.from_pretrained(tmp_dir)

        self.assertIsInstance(tokenizer2, tokenizer.__class__)
        self.assertEqual(tokenizer2.vocab_size, 12)

    def test_auto_tokenizer_fast_no_slow(self):
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl", from_hf_hub=True)
        # There is no fast CTRL so this always gives us a slow tokenizer.
        self.assertIsInstance(tokenizer, CTRLTokenizer)

    def test_get_tokenizer_config(self):
        # Check we can load the tokenizer config of an online model.
        config = get_tokenizer_config("google-bert/bert-base-cased")
        _ = config.pop("_commit_hash", None)
        # If we ever update google-bert/bert-base-cased tokenizer config, this dict here will need to be updated.
        self.assertEqual(config, {"do_lower_case": False, "model_max_length": 512})

        # This model does not have a tokenizer_config so we get back an empty dict.
        config = get_tokenizer_config(SMALL_MODEL_IDENTIFIER)
        self.assertDictEqual(config, {})

        # A tokenizer saved with `save_pretrained` always creates a tokenizer config.
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER, from_hf_hub=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            config = get_tokenizer_config(tmp_dir)

        # Check the class of the tokenizer was properly saved (note that it always saves the slow class).
        self.assertEqual(config["tokenizer_class"], "BertTokenizer")

    def test_new_tokenizer_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)

            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            # Trying to register something existing in the PaddleNLP library will raise an error
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, slow_tokenizer_class=BertTokenizer)

            tokenizer = CustomTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER, from_hf_hub=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir)

                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
                self.assertIsInstance(new_tokenizer, CustomTokenizer)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]

    @require_tokenizers
    def test_new_tokenizer_fast_registration(self):
        try:
            # Trying to register nothing
            with self.assertRaises(ValueError):
                AutoTokenizer.register(CustomConfig)
            # Trying to register tokenizer with wrong type
            with self.assertRaises(ValueError):
                AutoTokenizer.register(CustomConfig, fast_tokenizer_class=CustomTokenizer)
            with self.assertRaises(ValueError):
                AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizerFast)
            with self.assertRaises(ValueError):
                AutoTokenizer.register(
                    CustomConfig,
                    slow_tokenizer_class=CustomTokenizer,
                    fast_tokenizer_class=CustomTokenizerFastWithoutSlow,
                )
            AutoConfig.register("custom", CustomConfig)

            # Can register in two steps
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, None))
            AutoTokenizer.register(CustomConfig, fast_tokenizer_class=CustomTokenizerFast)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, CustomTokenizerFast))

            del TOKENIZER_MAPPING._extra_content[CustomConfig]
            # Can register in one step
            AutoTokenizer.register(
                CustomConfig, slow_tokenizer_class=CustomTokenizer, fast_tokenizer_class=CustomTokenizerFast
            )
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, CustomTokenizerFast))

            # Trying to register something existing in the PaddleNLP library will raise an error
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, fast_tokenizer_class=BertTokenizerFast)
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, slow_tokenizer_class=BertTokenizer)

            # We pass through a bert tokenizer fast cause there is no converter slow to fast for our new tokenizer
            # and that model does not have a tokenizer.json
            with tempfile.TemporaryDirectory() as tmp_dir:
                bert_tokenizer = BertTokenizerFast.from_pretrained(SMALL_MODEL_IDENTIFIER, from_hf_hub=True)
                bert_tokenizer.save_pretrained(tmp_dir)
                tokenizer = CustomTokenizerFast.from_pretrained(tmp_dir)

            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir, legacy_format=True)

                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, use_fast=True)
                self.assertIsInstance(new_tokenizer, CustomTokenizerFast)

                # TODO: fix this test. Now keep loaded tokenizer type
                # new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, use_fast=False)
                # self.assertIsInstance(new_tokenizer, CustomTokenizer)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_tokenizer(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/test_dynamic_tokenizer")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=False
            )

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True)
        self.assertTrue(tokenizer.special_attribute_present)

        # Test the dynamic module is loaded only once.
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True
        )
        self.assertIs(tokenizer.__class__, reloaded_tokenizer.__class__)

        # Test tokenizer can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True)
        self.assertTrue(reloaded_tokenizer.special_attribute_present)

        if is_tokenizers_available():
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")
            self.assertEqual(reloaded_tokenizer.__class__.__name__, "NewTokenizerFast")

            # Test we can also load the slow version
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True, use_fast=False
            )
            self.assertTrue(tokenizer.special_attribute_present)
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
            # Test tokenizer can be reloaded.
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir)
                reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True, use_fast=False)
            self.assertEqual(reloaded_tokenizer.__class__.__name__, "NewTokenizer")
            self.assertTrue(reloaded_tokenizer.special_attribute_present)
        else:
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
            self.assertEqual(reloaded_tokenizer.__class__.__name__, "NewTokenizer")

        # The tokenizer file is cached in the snapshot directory. So the module file is not changed after dumping
        # to a temp dir. Because the revision of the module file is not changed.
        # Test the dynamic module is loaded only once if the module file is not changed.
        self.assertIs(tokenizer.__class__, reloaded_tokenizer.__class__)

        # Test the dynamic module is reloaded if we force it.
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True, force_download=True
        )
        self.assertIsNot(tokenizer.__class__, reloaded_tokenizer.__class__)
        self.assertTrue(reloaded_tokenizer.special_attribute_present)

    @require_tokenizers
    def test_from_pretrained_dynamic_tokenizer_conflict(self):
        class NewTokenizer(BertTokenizer):
            special_attribute_present = False

        class NewTokenizerFast(BertTokenizerFast):
            slow_tokenizer_class = NewTokenizer
            special_attribute_present = False

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=NewTokenizer)
            AutoTokenizer.register(CustomConfig, fast_tokenizer_class=NewTokenizerFast)
            # If remote code is not set, the default is to use local
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/test_dynamic_tokenizer")
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")
            self.assertFalse(tokenizer.special_attribute_present)
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/test_dynamic_tokenizer", use_fast=False)
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
            self.assertFalse(tokenizer.special_attribute_present)

            # If remote code is disabled, we load the local one.
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=False
            )
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")
            self.assertFalse(tokenizer.special_attribute_present)
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=False, use_fast=False
            )
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
            self.assertFalse(tokenizer.special_attribute_present)

            # If remote is enabled, we load from the Hub
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True
            )
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")
            self.assertTrue(tokenizer.special_attribute_present)
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer", trust_remote_code=True, use_fast=False
            )
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
            self.assertTrue(tokenizer.special_attribute_present)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_tokenizer_legacy_format(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/test_dynamic_tokenizer_legacy", trust_remote_code=True
        )
        self.assertTrue(tokenizer.special_attribute_present)
        if is_tokenizers_available():
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")

            # Test we can also load the slow version
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/test_dynamic_tokenizer_legacy", trust_remote_code=True, use_fast=False
            )
            self.assertTrue(tokenizer.special_attribute_present)
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
        else:
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "bert-base is not a local folder and is not a valid model identifier"
        ):
            _ = AutoTokenizer.from_pretrained("bert-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoTokenizer.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_cached_tokenizer_has_minimum_calls_to_head(self):
        # Make sure we have cached the tokenizer.
        _ = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        with RequestCounter() as counter:
            _ = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        self.assertEqual(counter["GET"], 0)
        self.assertEqual(counter["HEAD"], 1)
        self.assertEqual(counter.total_calls, 1)
