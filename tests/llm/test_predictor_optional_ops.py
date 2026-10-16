# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm.predict import predictor


class OptionalPredictorOpsTest(unittest.TestCase):
    def test_predict_without_custom_ops(self):
        for options in ({}, {"inference_model": True}, {"block_attn": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output.json"
                data = Path(directory) / "input.json"
                data.write_text(json.dumps({"src": "hello", "tgt": "world"}) + "\n", encoding="utf-8")
                args = predictor.PredictorArgument(**options)
                model_args = predictor.ModelArgument(data_file=str(data), output_file=str(output))
                model = SimpleNamespace(tensor_parallel_rank=0, predict=lambda texts: ["answer"] * len(texts))
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            predictor.PdArgumentParser, "parse_args_into_dataclasses", return_value=(args, model_args)
                        )
                    )
                    stack.enter_context(patch.object(predictor.llm_utils, "set_triton_cache"))
                    load_ops = stack.enter_context(
                        patch("paddle.utils.try_import", side_effect=ImportError("missing custom operators"))
                    )
                    create = stack.enter_context(patch.object(predictor, "create_predictor", return_value=model))
                    predictor.predict()
                    if args.inference_model:
                        load_ops.assert_called_once_with("paddlenlp_ops")
                        create.assert_not_called()
                        self.assertFalse(output.exists())
                    else:
                        create.assert_called_once_with(args, model_args)
                        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["output"], "answer")
                        load_ops.assert_not_called()
