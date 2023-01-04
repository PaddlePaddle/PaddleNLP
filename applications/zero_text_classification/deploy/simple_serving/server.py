# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp import SimpleServer
from paddlenlp.prompt import PromptDataCollatorWithPadding, UTCTemplate
from paddlenlp.server import BaseModelHandler, MultiLabelClassificationPostHandler


class UTCModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, predictor, tokenizer, data, parameters):
        max_length = parameters.get("max_length", 128)
        batch_size = parameters.get("batch_size", 1)
        choices = parameters.get("choices", None)
        template = UTCTemplate(tokenizer, max_length)
        collator = PromptDataCollatorWithPadding(tokenizer, padding=True, return_tensors="np")

        for example in data:
            if "choices" not in example:
                example["choices"] = choices
            if "text_b" not in example:
                example["text_b"] = ""

        tokenized_data = [template(x) for x in data]
        batches = [[i, i + batch_size] for i in range(0, len(tokenized_data), batch_size)]
        results = [[] for i in range(0, predictor._output_num)]
        for start, end in batches:
            batch_data = collator(tokenized_data[start:end])
            for k, v in batch_data.items():
                if k == "attention_mask":
                    batch_data[k] = v.astype("float32")
                else:
                    batch_data[k] = v.astype("int64")
            if predictor._predictor_type == "paddle_inference":
                predictor._input_handles[0].copy_from_cpu(batch_data["input_ids"])
                predictor._input_handles[1].copy_from_cpu(batch_data["token_type_ids"])
                predictor._input_handles[2].copy_from_cpu(batch_data["position_ids"])
                predictor._input_handles[3].copy_from_cpu(batch_data["attention_mask"])
                predictor._input_handles[4].copy_from_cpu(batch_data["omask_positions"])
                predictor._input_handles[5].copy_from_cpu(batch_data["cls_positions"])
                predictor._predictor.run()
                output = [output_handle.copy_to_cpu() for output_handle in predictor._output_handles]
            else:
                output = predictor._predictor.run(None, batch_data)
            for i, out in enumerate(output):
                results[i].extend(out.tolist())

        out_dict = {"logits": results[0], "data": data}
        for i in range(1, len(results)):
            out_dict[f"logits_{i}"] = results[i]
        return out_dict


app = SimpleServer()
app.register(
    "models/utc",
    model_path="../../checkpoint/model_best/",
    tokenizer_name="utc-large",
    model_handler=UTCModelHandler,
    post_handler=MultiLabelClassificationPostHandler,
)
