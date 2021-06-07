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
        prediction_probs = model(
            input_ids=src_ids, token_type_ids=token_type_ids).numpy()

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


@paddle.no_grad()
def do_evaluate_chid(model, tokenizer, data_loader, label_normalize_dict):
    """
        FewCLUE `chid` dataset is specical when evaluate: input slots have 
        additional `candidate_label_ids`, so need to customize the
        evaluate function.
    """

    model.eval()

    total_num = 0
    correct_num = 0

    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]

    label_length = len(normed_labels[0])

    for batch in data_loader:
        src_ids, token_type_ids, masked_positions, masked_lm_labels, candidate_label_ids = batch

        # [bs * label_length, vocab_size]
        prediction_probs = model.predict(
            input_ids=src_ids,
            token_type_ids=token_type_ids,
            masked_positions=masked_positions)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]

        # prediction_probs: [batch_size, label_lenght, vocab_size]
        prediction_probs = paddle.reshape(
            prediction_probs, shape=[batch_size, -1, vocab_size]).numpy()

        candidate_num = candidate_label_ids.shape[1]

        # [batch_size, candidate_num(7)]
        y_pred = np.ones(shape=[batch_size, candidate_num])

        for label_idx in range(candidate_num):

            # [bathc_size, label_length(4)] 
            single_candidate_label_ids = candidate_label_ids[:, label_idx, :]
            # Calculate joint distribution of candidate labels
            for index in range(label_length):
                # [batch_size,]
                slice_word_ids = single_candidate_label_ids[:, index].numpy()

                batch_single_token_prob = []
                for bs_index in range(batch_size):
                    # [1, 1]
                    single_token_prob = prediction_probs[
                        bs_index, index, slice_word_ids[bs_index]]
                    batch_single_token_prob.append(single_token_prob)

                y_pred[:, label_idx] *= np.array(batch_single_token_prob)

        # Get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)

        y_true_index = []

        for index, masked_lm_label in enumerate(masked_lm_labels.numpy()):
            # [cantidate_num, label_length]
            tmp_candidate_label_ids = candidate_label_ids[index, :, :]
            for idx, label_ids in enumerate(tmp_candidate_label_ids.numpy()):
                if np.equal(label_ids, masked_lm_label).all():
                    y_true_index.append(idx)
                    continue

        y_true_index = np.array(y_true_index)

        total_num += len(y_true_index)
        correct_num += (y_true_index == y_pred_index).sum()

    return 100 * correct_num / total_num, total_num
