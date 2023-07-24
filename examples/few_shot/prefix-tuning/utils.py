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

import numpy as np
import paddle
from rouge import Rouge

from paddlenlp.metrics import BLEU
from paddlenlp.prompt import PromptTrainer


# Define the metric function.
def compute_metrics(eval_preds, tokenizer):

    all_preds = []
    all_labels = []
    labels = eval_preds.label_ids
    preds = eval_preds.predictions
    all_preds.extend(tokenizer.convert_ids_to_string(pred.tolist()) for pred in preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    all_labels.extend(tokenizer.convert_ids_to_string(label.tolist()) for label in labels)

    assert len(all_preds) == len(all_labels), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(all_preds), len(all_labels))
    )
    rouge = Rouge()
    bleu4 = BLEU(n_size=4)
    scores = []
    for pred, target in zip(all_preds, all_labels):
        try:
            score = rouge.get_scores(" ".join(pred), " ".join(target))
            scores.append([score[0]["rouge-1"]["f"], score[0]["rouge-2"]["f"], score[0]["rouge-l"]["f"]])
        except ValueError:
            scores.append([0, 0, 0])
        bleu4.add_inst(pred, [target])
    rouge1 = np.mean([i[0] for i in scores])
    rouge2 = np.mean([i[1] for i in scores])
    rougel = np.mean([i[2] for i in scores])
    print("\n" + "*" * 15)
    print("The auto evaluation result is:")
    print("rouge-1:", round(rouge1, 4))
    print("rouge-2:", round(rouge2, 4))
    print("rouge-L:", round(rougel, 4))
    print("BLEU-4:", round(bleu4.score(), 4))
    return {"rougel": rougel}


class PromptTrainerForGeneration(PromptTrainer):
    def __init__(
        self,
        model,
        tokenizer,
        criterion=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
    ):
        super(PromptTrainerForGeneration, self).__init__(
            model=model,
            criterion=criterion,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.verbalizer = None

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to evaluate.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[paddle.Tensor], Optional[paddle.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        labels = inputs["labels"]
        # inputs = self._prepare_inputs(inputs)

        max_length = 32
        generated_tokens = self.model.generate(
            model_kwargs=inputs,
        )

        # different from hf returns: tuple[Tensor]: It is a tuple contains two elements: ids and scores.
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        # in case the batch is shorter than max length, the output should be padded
        if max_length is not None and generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        # paddle.ones need to support device args.
        padded_tensor = pad_token_id * paddle.ones((tensor.shape[0], max_length), dtype=tensor.dtype)
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
