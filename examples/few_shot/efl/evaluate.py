# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


@paddle.no_grad()
def do_evaluate(model, tokenizer, data_loader, task_label_description):
    model.eval()

    total_num = 0
    correct_num = 0

    class_num = len(task_label_description)

    # [total_num * class_num, 2]
    all_prediction_probs = []
    # [total_num * class_num]
    all_labels = []

    for batch in data_loader:
        src_ids, token_type_ids, true_labels = batch
        # Prediction_probs:[bs * class_num, 2]
        prediction_probs = model(input_ids=src_ids,
                                 token_type_ids=token_type_ids).numpy()

        all_prediction_probs.append(prediction_probs)
        all_labels.append(true_labels.numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_prediction_probs = np.concatenate(all_prediction_probs, axis=0)
    all_prediction_probs = np.reshape(all_prediction_probs, (-1, class_num, 2))

    prediction_pos_probs = all_prediction_probs[:, :, 1]
    prediction_pos_probs = np.reshape(prediction_pos_probs, (-1, class_num))
    y_pred_index = np.argmax(prediction_pos_probs, axis=-1)

    y_true_index = np.array([
        true_label_index for idx, true_label_index in enumerate(all_labels)
        if idx % class_num == 0
    ])

    total_num = len(y_true_index)
    correct_num = (y_pred_index == y_true_index).sum()

    model.train()
    return 100 * correct_num / total_num, total_num
