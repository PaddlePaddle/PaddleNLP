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

from utils import GLMTrainer, read_data

from paddlenlp.data import DefaultDataCollator
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import Rouge1, Rouge2, RougeL
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer


def main():
    parser = PdArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()

    train_ds = load_dataset(read_data, filenames=["26.json", "25.json"], lazy=False)
    dev_ds = load_dataset(read_data, filenames=["1.json"], lazy=False)

    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-large-chinese")
    model = AutoModelForConditionalGeneration.from_pretrained("THUDM/glm-large-chinese")

    collate_fn = DefaultDataCollator()

    def compute_metrics(eval_preds):
        rouge1 = Rouge1()
        rouge2 = Rouge2()
        rougel = RougeL()
        predictions = [x[x != -100] for x in eval_preds.predictions]
        references = [x[x != -100] for x in eval_preds.label_ids]

        rouge1_score = rouge1.score(predictions, references)
        rouge2_score = rouge2.score(predictions, references)
        for pred, ref in zip(predictions, references):
            rougel.add_inst(pred, [ref])
        return {
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougel": rougel.score(),
        }

    trainer = GLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        do_generation=True,
        data_collator=collate_fn,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
