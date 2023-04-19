# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import tempfile
import unittest

import paddle

from paddlenlp.transformers import BertModel
from paddlenlp.transformers.model_utils import shard_checkpoint
from paddlenlp.transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
)
from paddlenlp.utils.env import PADDLE_WEIGHT_FILE_NAME
from tests.testing_utils import require_package

# paddle.set_device("cpu")


class TestCkptShard(unittest.TestCase):
    def test_shard_checkpoint(self):
        # This is the model we will use, total size 340,000 bytes.
        model = paddle.nn.Sequential(
            paddle.nn.Linear(100, 200, bias_attr=False),  # size 80,000
            paddle.nn.Linear(200, 200, bias_attr=False),  # size 160,000
            paddle.nn.Linear(200, 100, bias_attr=False),  # size 80,000
            paddle.nn.Linear(100, 50, bias_attr=False),  # size 20,000
        )
        state_dict = model.state_dict()

        with self.subTest("No shard when max size is bigger than model size"):
            shards, index = shard_checkpoint(state_dict)
            self.assertIsNone(index)
            self.assertDictEqual(shards, {PADDLE_WEIGHT_FILE_NAME: state_dict})

        with self.subTest("Test sharding, no weights bigger than max size"):
            shards, index = shard_checkpoint(state_dict, max_shard_size="300kB")
            # Split is first two layers then last two.
            self.assertDictEqual(
                index,
                {
                    "metadata": {"total_size": 340000},
                    "weight_map": {
                        "0.weight": "model_state-00001-of-00002.pdparams",
                        "1.weight": "model_state-00001-of-00002.pdparams",
                        "2.weight": "model_state-00002-of-00002.pdparams",
                        "3.weight": "model_state-00002-of-00002.pdparams",
                    },
                },
            )

            shard1 = {"0.weight": state_dict["0.weight"], "1.weight": state_dict["1.weight"]}
            shard2 = {"2.weight": state_dict["2.weight"], "3.weight": state_dict["3.weight"]}
            self.assertDictEqual(
                shards, {"model_state-00001-of-00002.pdparams": shard1, "model_state-00002-of-00002.pdparams": shard2}
            )

        with self.subTest("Test sharding with weights bigger than max size"):
            shards, index = shard_checkpoint(state_dict, max_shard_size="100kB")
            # Split is first layer, second layer then last 2.
            import pdb

            pdb.set_trace()
            self.assertDictEqual(
                index,
                {
                    "metadata": {"total_size": 340000},
                    "weight_map": {
                        "0.weight": "model_state-00001-of-00003.pdparams",
                        "1.weight": "model_state-00002-of-00003.pdparams",
                        "2.weight": "model_state-00003-of-00003.pdparams",
                        "3.weight": "model_state-00003-of-00003.pdparams",
                    },
                },
            )

            shard1 = {"0.weight": state_dict["0.weight"]}
            shard2 = {"1.weight": state_dict["1.weight"]}
            shard3 = {"2.weight": state_dict["2.weight"], "3.weight": state_dict["3.weight"]}
            self.assertDictEqual(
                shards,
                {
                    "model_state-00001-of-00003.pdparams": shard1,
                    "model_state-00002-of-00003.pdparams": shard2,
                    "model_state-00003-of-00003.pdparams": shard3,
                },
            )

    def test_checkpoint_sharding_local(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # We use the same folder for various sizes to make sure a new save erases the old checkpoint.
            for max_size in ["50kB", "50kiB", "100kB", "100kiB", "200kB", "200kiB"]:
                model.save_pretrained(tmp_dir, max_shard_size=max_size)

                # Get each shard file and its size
                shard_to_size = {}
                for shard in os.listdir(tmp_dir):
                    if shard.endswith(".pdparams"):
                        shard_file = os.path.join(tmp_dir, shard)
                        shard_to_size[shard_file] = os.path.getsize(shard_file)

                index_file = os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)
                # Check there is an index but no regular weight file
                self.assertTrue(os.path.isfile(index_file))
                self.assertFalse(os.path.isfile(os.path.join(tmp_dir, PADDLE_WEIGHT_FILE_NAME)))

                # Check a file is bigger than max_size only when it has a single weight
                for shard_file, size in shard_to_size.items():
                    if max_size.endswith("kiB"):
                        max_size_int = int(max_size[:-3]) * 2**10
                    else:
                        max_size_int = int(max_size[:-2]) * 10**3
                    # Note: pickle adds some junk so the weight of the file can end up being slightly bigger than
                    # the size asked for (since we count parameters)
                    if size >= max_size_int + 50000:
                        state_dict = paddle.load(shard_file)
                        self.assertEqual(len(state_dict), 1)

                # Check the index and the shard files found match
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.loads(f.read())

                all_shards = set(index["weight_map"].values())
                shards_found = {f for f in os.listdir(tmp_dir) if f.endswith(".pdparams")}
                self.assertSetEqual(all_shards, shards_found)

                # Finally, check the model can be reloaded
                new_model = BertModel.from_pretrained(tmp_dir)
                for p1, p2 in zip(model.parameters(), new_model.parameters()):
                    self.assertTrue(paddle.allclose(p1, p2))

    def test_checkpoint_sharding_from_hub(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded", from_hf_hub=True)
        # the model above is the same as the model below, just a sharded version.
        ref_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        for p1, p2 in zip(model.parameters(), ref_model.parameters()):
            self.assertTrue(paddle.allclose(p1, p2))

    def test_checkpoint_variant_local(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2")

            weights_name = ".".join(PADDLE_WEIGHT_FILE_NAME.split(".")[:-1] + ["v2"] + ["bin"])

            weights_file = os.path.join(tmp_dir, weights_name)
            self.assertTrue(os.path.isfile(weights_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, PADDLE_WEIGHT_FILE_NAME)))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(paddle.allclose(p1, p2))

    def test_checkpoint_variant_local_sharded(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", max_shard_size="50kB")

            weights_index_name = ".".join(WEIGHTS_INDEX_NAME.split(".")[:-1] + ["v2"] + ["json"])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)))

            for i in range(1, 6):
                weights_name = ".".join(PADDLE_WEIGHT_FILE_NAME.split(".")[:-1] + [f"v2-0000{i}-of-00006"] + ["bin"])
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(paddle.allclose(p1, p2))

    @require_package("safetensors")
    def test_checkpoint_variant_local_safe(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", safe_serialization=True)

            weights_name = ".".join(SAFE_WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["safetensors"])

            weights_file = os.path.join(tmp_dir, weights_name)
            self.assertTrue(os.path.isfile(weights_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(paddle.allclose(p1, p2))

    @require_package("safetensors")
    def test_checkpoint_variant_local_sharded_safe(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", max_shard_size="50kB", safe_serialization=True)

            weights_index_name = ".".join(SAFE_WEIGHTS_INDEX_NAME.split(".")[:-1] + ["v2"] + ["json"])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))

            for i in range(1, 6):
                weights_name = ".".join(SAFE_WEIGHTS_NAME.split(".")[:-1] + [f"v2-0000{i}-of-00006"] + ["safetensors"])
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(paddle.allclose(p1, p2))

    def test_checkpoint_variant_hub(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir)
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    def test_checkpoint_variant_hub_sharded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(
                    "hf-internal-testing/tiny-random-bert-variant-sharded", cache_dir=tmp_dir
                )
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-sharded", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    @require_package("safetensors")
    def test_checkpoint_variant_hub_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-variant-safe", cache_dir=tmp_dir)
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-safe", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    @require_package("safetensors")
    def test_checkpoint_variant_hub_sharded_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(
                    "hf-internal-testing/tiny-random-bert-variant-sharded-safe", cache_dir=tmp_dir
                )
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-sharded-safe", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    def test_checkpoint_variant_save_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir, variant="v2"
            )
            weights_name = ".".join(PADDLE_WEIGHT_FILE_NAME.split(".")[:-1] + ["v2"] + ["bin"])

            model.save_pretrained(tmp_dir, variant="v2")
            # saving will create a variant checkpoint
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, weights_name)))

            model.save_pretrained(tmp_dir)
            # saving shouldn't delete variant checkpoints
            weights_name = ".".join(PADDLE_WEIGHT_FILE_NAME.split(".")[:-1] + ["v2"] + ["bin"])
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, weights_name)))

            # there should be a normal checkpoint
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, PADDLE_WEIGHT_FILE_NAME)))

        self.assertIsNotNone(model)
