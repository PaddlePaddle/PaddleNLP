# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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

from typing import List, Optional, Dict, Any, Union, Callable

import logging
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from collections import defaultdict
from time import perf_counter

import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModelForQuestionAnswering, AutoTokenizer
from paddlenlp.data import Stack, Tuple

from pipelines.data_handler.samples import SampleBasket
from pipelines.data_handler.processor import SquadProcessor, Processor
from pipelines.data_handler.inputs import QAInput, Question
from pipelines.data_handler.predictions import QAPred, QACandidate
from pipelines.utils.common_utils import initialize_device_settings, try_get

from pipelines.schema import Document, Answer, Span
from pipelines.document_stores import BaseDocumentStore
from pipelines.nodes.reader import BaseReader

logger = logging.getLogger(__name__)


class ErnieReader(BaseReader):
    """
    Transformer based model for extractive Question Answering based on ERNIE3.0.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_version: Optional[str] = None,
        context_window_size: int = 150,
        batch_size: int = 50,
        use_gpu: bool = True,
        no_ans_boost: float = 0.0,
        return_no_answer: bool = False,
        top_k: int = 10,
        top_k_per_candidate: int = 3,
        top_k_per_sample: int = 1,
        num_processes: Optional[int] = None,
        max_seq_len: int = 256,
        doc_stride: int = 128,
        progress_bar: bool = True,
        duplicate_filtering: int = 0,
        use_confidence_scores: bool = True,
        proxies: Optional[Dict[str, str]] = None,
        local_files_only=False,
        force_download=False,
        use_auth_token: Optional[Union[str, bool]] = None,
        n_best_per_sample: int = 1,
        use_confidence_scores_for_ranking: bool = False,
        n_best: int = 5,
        **kwargs,
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'ernie-gram-zh-finetuned-dureader-robust'.
        :param context_window_size: The size, in characters, of the window around the answer span that is used when
                                    displaying the context around the answer.
        :param batch_size: Number of samples the model receives in one batch for inference.
                           Memory consumption is much lower in inference mode. Recommendation: Increase the batch size
                           to a value so only a single batch is used.
        :param use_gpu: Whether to use GPU (if available)
        :param no_ans_boost: How much the no_answer logit is boosted/increased.
        If set to 0 (default), the no_answer logit is not changed.
        If a negative number, there is a lower chance of "no_answer" being predicted.
        If a positive number, there is an increased chance of "no_answer"
        :param return_no_answer: Whether to include no_answer predictions in the results.
        :param top_k: The maximum number of answers to return
        :param top_k_per_candidate: How many answers to extract for each candidate doc that is coming from the retriever (might be a long text).
        Note that this is not the number of "final answers" you will receive
        (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
        and that FARM includes no_answer in the sorted list of predictions.
        :param top_k_per_sample: How many answers to extract from each small text passage that the model can process at once
        (one "candidate doc" is usually split into many smaller "passages").
        You usually want a very small value here, as it slows down inference
        and you don't gain much of quality by having multiple answers from one passage.
        Note that this is not the number of "final answers" you will receive
        (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
        and that FARM includes no_answer in the sorted list of predictions.
        :param num_processes: The number of processes for `multiprocessing.Pool`. Set to value of 0 to disable
                              multiprocessing. Set to None to let Inferencer determine optimum number. If you
                              want to debug the Language Model, you might need to disable multiprocessing!
        :param max_seq_len: Max sequence length of one input text for the model
        :param doc_stride: Length of striding window for splitting long texts (used if ``len(text) > max_seq_len``)
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_filtering: Answers are filtered based on their position. Both start and end position of the answers are considered.
                                    The higher the value, answers that are more apart are filtered out. 0 corresponds to exact duplicates. -1 turns off duplicate removal.
        :param use_confidence_scores: Sets the type of score that is returned with every predicted answer.
                                      `True` => a scaled confidence / relevance score between [0, 1].
                                      This score can also be further calibrated on your dataset via self.eval()
                                      `False` => an unscaled, raw score [-inf, +inf] which is the sum of start and end logit
                                      from the model for the predicted span.
        :param proxies: Dict of proxy servers to use for downloading external models. Example: {'http': 'some.proxy:1234', 'http://hostname': 'my.proxy:3111'}
        :param local_files_only: Whether to force checking for local files only (and forbid downloads)
        :param force_download: Whether fo force a (re-)download even if the model exists locally in the cache.
        :param n_best: The number of positive answer spans for each document.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path,
            context_window_size=context_window_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
            no_ans_boost=no_ans_boost,
            return_no_answer=return_no_answer,
            top_k=top_k,
            top_k_per_candidate=top_k_per_candidate,
            top_k_per_sample=top_k_per_sample,
            num_processes=num_processes,
            max_seq_len=max_seq_len,
            doc_stride=doc_stride,
            progress_bar=progress_bar,
            duplicate_filtering=duplicate_filtering,
            proxies=proxies,
            local_files_only=local_files_only,
            force_download=force_download,
            use_confidence_scores=use_confidence_scores,
            **kwargs,
        )

        self.batch_size = batch_size
        self.devices, _ = initialize_device_settings(use_cuda=use_gpu,
                                                     multi_gpu=False)

        self.return_no_answers = return_no_answer
        self.top_k = top_k
        self.top_k_per_candidate = top_k_per_candidate

        # Add by tianxin04
        self.n_best_per_sample = n_best_per_sample
        self.duplicate_filtering = duplicate_filtering
        self.no_ans_boost = no_ans_boost
        self.use_confidence_scores_for_ranking = use_confidence_scores_for_ranking
        self.n_best = n_best
        self.context_window_size = context_window_size

        # load_model
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path)
        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.processor = SquadProcessor(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            label_list=["start_token", "end_token"],
            metric="squad",
            data_dir="data",
            doc_stride=doc_stride,
        )

        self.max_seq_len = max_seq_len
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar
        self.use_confidence_scores = use_confidence_scores

    def predict(self,
                query: str,
                documents: List[Document],
                top_k: Optional[int] = None):
        """
        Use loaded QA model to find answers for a query in the supplied list of Document.

        Returns dictionaries containing answers sorted by (desc.) score.
        Example:
         ```python
            |{
            |    'query': 'Who is the father of Arya Stark?',
            |    'answers':[Answer(
            |                 'answer': 'Eddard,',
            |                 'context': "She travels with her father, Eddard, to King's Landing when he is",
            |                 'score': 0.9787139466668613,
            |                 'offsets_in_context': [Span(start=29, end=35],
            |                 'offsets_in_context': [Span(start=347, end=353],
            |                 'document_id': '88d1ed769d003939d3a0d28034464ab2'
            |                 ),...
            |              ]
            |}
         ```

        :param query: Query string
        :param documents: List of Document in which to search for the answer
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers
        """
        if top_k is None:
            top_k = self.top_k
        # convert input to FARM format
        inputs = []
        for doc in documents:
            # QAInput Class
            cur = QAInput(doc_text=doc.content,
                          questions=Question(text=query, uid=doc.id))
            inputs.append(cur)

        # get answers from QA model
        # TODO: Need fix in FARM's `to_dict` function of `QAInput` class

        # convert Document to dicts
        dicts = [o.to_dict() for o in inputs]

        # Generate dataset
        indices = list(range(len(dicts)))
        dataset, tensor_names, problematic_ids, baskets = self.processor.dataset_from_dicts(
            dicts, indices=indices, return_baskets=True)

        # Need more elegent implementation
        self.baskets = baskets

        predictions = self._get_predictions_and_aggregate(
            dataset, tensor_names, baskets)

        # assemble answers from all the different documents & format them.
        answers, max_no_ans_gap = self._extract_answers_of_predictions(
            predictions, top_k)
        # TODO: potentially simplify return here to List[Answer] and handle no_ans_gap differently
        result = {
            "query": query,
            "no_ans_gap": max_no_ans_gap,
            "answers": answers
        }

        return result

    def _extract_answers_of_predictions(self,
                                        predictions: List[QAPred],
                                        top_k: Optional[int] = None):
        # Assemble answers from all the different documents and format them.
        # For the 'no answer' option, we collect all no_ans_gaps and decide how likely
        # a no answer is based on all no_ans_gaps values across all documents
        answers: List[Answer] = []
        no_ans_gaps = []
        best_score_answer = 0

        for pred in predictions:
            answers_per_document = []
            no_ans_gaps.append(pred.no_answer_gap)
            for ans in pred.prediction:
                # skip 'no answers' here
                if self._check_no_answer(ans):
                    pass
                else:
                    cur = Answer(
                        answer=ans.answer,
                        type="extractive",
                        score=ans.confidence
                        if self.use_confidence_scores else ans.score,
                        context=ans.context_window,
                        document_id=pred.id,
                        offsets_in_context=[
                            Span(
                                start=ans.offset_answer_start -
                                ans.offset_context_window_start,
                                end=ans.offset_answer_end -
                                ans.offset_context_window_start,
                            )
                        ],
                        offsets_in_document=[
                            Span(start=ans.offset_answer_start,
                                 end=ans.offset_answer_end)
                        ],
                    )

                    answers_per_document.append(cur)

                    if ans.score > best_score_answer:
                        best_score_answer = ans.score

            # Only take n best candidates. Answers coming back from FARM are sorted with decreasing relevance
            answers += answers_per_document[:self.top_k_per_candidate]

        # calculate the score for predicting 'no answer', relative to our best positive answer score
        no_ans_prediction, max_no_ans_gap = self._calc_no_answer(
            no_ans_gaps, best_score_answer, self.use_confidence_scores)
        if self.return_no_answers:
            answers.append(no_ans_prediction)

        # sort answers by score (descending) and select top-k
        answers = sorted(answers, reverse=True)
        answers = answers[:top_k]

        return answers, max_no_ans_gap

    def calibrate_confidence_scores(
        self,
        document_store: BaseDocumentStore,
        device: Optional[str] = None,
        label_index: str = "label",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
    ):
        """
        Calibrates confidence scores on evaluation documents in the DocumentStore.

        :param document_store: DocumentStore containing the evaluation documents
        :param device: The device on which the tensors should be processed. Choose from "cpu" and "cuda" or use the Reader's device by default.
        :param label_index: Index/Table name where labeled questions are stored
        :param doc_index: Index/Table name where documents that are used for evaluation are stored
        :param label_origin: Field name where the gold labels are stored
        """
        if device is None:
            device = self.devices[0]
        self.eval(
            document_store=document_store,
            device=device,
            label_index=label_index,
            doc_index=doc_index,
            label_origin=label_origin,
            calibrate_conf_scores=True,
        )

    @staticmethod
    def _check_no_answer(c: QACandidate):
        # check for correct value in "answer"
        if c.offset_answer_start == 0 and c.offset_answer_end == 0:
            if c.answer != "no_answer":
                logger.error(
                    "Invalid 'no_answer': Got a prediction for position 0, but answer string is not 'no_answer'"
                )
        return c.answer == "no_answer"

    def predict_on_texts(self,
                         question: str,
                         texts: List[str],
                         top_k: Optional[int] = None):
        """
        Use loaded QA model to find answers for a question in the supplied list of Document.
        Returns dictionaries containing answers sorted by (desc.) score.
        Example:
         ```python
            |{
            |    'question': 'Who is the father of Arya Stark?',
            |    'answers':[
            |                 {'answer': 'Eddard,',
            |                 'context': " She travels with her father, Eddard, to King's Landing when he is ",
            |                 'offset_answer_start': 147,
            |                 'offset_answer_end': 154,
            |                 'score': 0.9787139466668613,
            |                 'document_id': '1337'
            |                 },...
            |              ]
            |}
         ```

        :param question: Question string
        :param documents: List of documents as string type
        :param top_k: The maximum number of answers to return
        :return: Dict containing question and answers
        """
        documents = []
        for text in texts:
            documents.append(Document(content=text))
        predictions = self.predict(question, documents, top_k)
        return predictions

    def _get_predictions_and_aggregate(self, dataset, tensor_names: List,
                                       baskets: List[SampleBasket]):
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + logits_to_preds + formatted_preds).

        Difference to _get_predictions():
         - Additional aggregation step across predictions of individual samples
         (e.g. For QA on long texts, we extract answers from multiple passages and then aggregate them on the "document level")

        :param dataset: Paddle Dataset with samples you want to predict
        :param tensor_names: Names of the tensors in the dataset
        :param baskets: For each item in the dataset, we need additional information to create formatted preds.
                        Baskets contain all relevant infos for that.
                        Example: QA - input string to convert the predicted answer from indices back to string space
        :return: list of predictions
        """

        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=self.batch_size,
                                               shuffle=False)

        batchify_fn = lambda samples, fn=Tuple(
            Stack(dtype="int64"),  # input_ids
            Stack(dtype="int64"),  # input_ids
            Stack(dtype="int64"),  # input_ids
            Stack(dtype="int64"),  # input_ids
            Stack(dtype="int64"),  # input_ids
            Stack(dtype="int64"),  # input_ids
            Stack(dtype="int64"),  # input_ids
            Stack(dtype="int64"),  # input_ids
            Stack(dtype="int64"),  # input_ids
        ): [data for data in fn(samples)]

        data_loader = paddle.io.DataLoader(dataset=dataset,
                                           batch_sampler=batch_sampler,
                                           collate_fn=batchify_fn,
                                           return_list=True)

        # TODO Sometimes this is the preds of one head, sometimes of two. We need a more advanced stacking operation
        # TODO so that preds of the right shape are passed in to formatted_preds
        unaggregated_preds_all = []

        # for i, batch in enumerate(
        #     tqdm(data_loader, desc=f"Inferencing Samples", unit=" Batches", disable=False)
        # ):
        for i, batch in enumerate(data_loader):

            (input_ids, padding_mask, segment_ids, passage_start_t,
             start_of_word, labels, id, seq_2_start_t, span_mask) = batch

            # get logits
            with paddle.no_grad():
                # Aggregation works on preds, not logits. We want as much processing happening in one batch + on GPU
                # So we transform logits to preds here as well
                start_logits, end_logits = self.model.forward(
                    input_ids=input_ids, token_type_ids=segment_ids)
                start_logits = paddle.unsqueeze(start_logits, axis=2)
                end_logits = paddle.unsqueeze(end_logits, axis=2)
                logits = paddle.concat(x=[start_logits, end_logits], axis=-1)

                preds = self.logits_to_preds(logits,
                                             span_mask=span_mask,
                                             start_of_word=start_of_word,
                                             seq_2_start_t=seq_2_start_t)

                unaggregated_preds_all.append(preds)

        # In some use cases we want to aggregate the individual predictions.
        # This is mostly useful, if the input text is longer than the max_seq_len that the model can process.
        # In QA we can use this to get answers from long input texts by first getting predictions for smaller passages
        # and then aggregating them here.

        # At this point unaggregated preds has shape [n_batches][n_heads][n_samples]

        # can assume that we have only complete docs i.e. all the samples of one doc are in the current chunk
        logits = [None]
        preds_all = self.formatted_preds_wrapper(
            logits=
            logits,  # For QA we collected preds per batch and do not want to pass logits
            preds=unaggregated_preds_all,
            baskets=self.baskets,
        )  # type ignore
        return preds_all

    def logits_to_preds(
        self,
        logits: paddle.Tensor,
        span_mask: paddle.Tensor,
        start_of_word: paddle.Tensor,
        seq_2_start_t: paddle.Tensor,
        max_answer_length: int = 1000,
        **kwargs,
    ):
        """
        Get the predicted index of start and end token of the answer. Note that the output is at token level
        and not word level. Note also that these logits correspond to the tokens of a sample
        (i.e. special tokens, question tokens, passage_tokens)
        """

        # Will be populated with the top-n predictions of each sample in the batch
        # shape = batch_size x ~top_n
        # Note that ~top_n = n   if no_answer is     within the top_n predictions
        #           ~top_n = n+1 if no_answer is not within the top_n predictions
        all_top_n = []

        # logits is of shape [batch_size, max_seq_len, 2]. The final dimension corresponds to [start, end]
        start_logits, end_logits = paddle.split(logits,
                                                num_or_sections=2,
                                                axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Calculate a few useful variables
        batch_size = start_logits.shape[0]
        max_seq_len = start_logits.shape[1]  # target dim

        # get scores for all combinations of start and end logits => candidate answers
        # [22, 256] -> [22, 256, 1] -> [22, 256, 256]
        start_matrix = paddle.expand(start_logits.unsqueeze(2),
                                     shape=[-1, -1, max_seq_len])
        # [22, 256] -> [22, 1, 256] -> [22, 256, 256]
        end_matrix = paddle.expand(end_logits.unsqueeze(1),
                                   shape=[-1, max_seq_len, -1])
        start_end_matrix = start_matrix + end_matrix

        # disqualify answers where end < start
        # (set the lower triangular matrix to low value, excluding diagonal)
        # The answer positions that end position less than start position shuold be mask
        pos_mask_tensor = paddle.tensor.triu((paddle.ones(
            (max_seq_len, max_seq_len), dtype=paddle.get_default_dtype()) *
                                              -888),
                                             diagonal=1)
        pos_mask_tensor = paddle.transpose(pos_mask_tensor, perm=[1, 0])

        masked_start_end_matrix = []
        for single_start_end_matrix in start_end_matrix:
            single_start_end_matrix += pos_mask_tensor
            masked_start_end_matrix.append(
                paddle.unsqueeze(single_start_end_matrix, axis=0))
        start_end_matrix = paddle.concat(x=masked_start_end_matrix, axis=0)

        # Todo(tianxin04): mask long span
        # disqualify answers where answer span is greater than max_answer_length
        # (set the upper triangular matrix to low value, excluding diagonal)
        # indices_long_span = torch.triu_indices(
        #     max_seq_len, max_seq_len, offset=max_answer_length, device=start_end_matrix.device
        # )
        # start_end_matrix[:, indices_long_span[0][:], indices_long_span[1][:]] = -777

        # disqualify answers where start=0, but end != 0
        start_end_matrix[:, 0, 1:] = -666

        # Turn 1d span_mask vectors into 2d span_mask along 2 different axes
        # span mask has:
        #   0 for every position that is never a valid start or end index (question tokens, mid and end special tokens, padding)
        #   1 everywhere else
        # [22, 256] -> [22, 256, 1] -> [22, 256, 256]
        span_mask_start = paddle.expand(paddle.unsqueeze(span_mask, axis=2),
                                        shape=[-1, -1, max_seq_len])
        span_mask_end = paddle.expand(paddle.unsqueeze(span_mask, axis=1),
                                      shape=[-1, max_seq_len, -1])
        span_mask_2d = span_mask_start + span_mask_end

        # disqualify spans where either start or end is on an invalid token
        invalid_indices = paddle.nonzero((span_mask_2d != 2), as_tuple=True)
        # Todo(tianxin04):
        # Hack: This Paddle operation is very time consuming, so convert Paddle.Tensor to numpy.array
        # and then convert back to Paddle.Tensor
        start_end_matrix = start_end_matrix.numpy()
        start_end_matrix[invalid_indices[0][:], invalid_indices[1][:],
                         invalid_indices[2][:]] = -999
        start_end_matrix = paddle.to_tensor(start_end_matrix,
                                            place=self.devices[0])

        # Sort the candidate answers by their score. Sorting happens on the flattened matrix.
        # flat_sorted_indices.shape: (batch_size, max_seq_len^2, 1)
        flat_scores = paddle.reshape(start_end_matrix, shape=[batch_size, -1])
        flat_sorted_indices_2d = paddle.argsort(flat_scores,
                                                axis=-1,
                                                descending=True)
        flat_sorted_indices = paddle.unsqueeze(flat_sorted_indices_2d, axis=2)

        # The returned indices are then converted back to the original dimensionality of the matrix.
        # sorted_candidates.shape : (batch_size, max_seq_len^2, 2)
        start_indices = flat_sorted_indices // max_seq_len
        end_indices = flat_sorted_indices % max_seq_len
        sorted_candidates = paddle.concat(x=[start_indices, end_indices],
                                          axis=2)

        # Get the n_best candidate answers for each sample
        for sample_idx in range(batch_size):
            sample_top_n = self.get_top_candidates(
                sorted_candidates[sample_idx],
                start_end_matrix[sample_idx],
                sample_idx,
                start_matrix=start_matrix[sample_idx],
                end_matrix=end_matrix[sample_idx],
            )
            all_top_n.append(sample_top_n)

        return all_top_n

    def get_top_candidates(self, sorted_candidates, start_end_matrix,
                           sample_idx: int, start_matrix, end_matrix):
        """
        Returns top candidate answers as a list of Span objects. Operates on a matrix of summed start and end logits.
        This matrix corresponds to a single sample (includes special tokens, question tokens, passage tokens).
        This method always returns a list of len n_best_per_sample + 1 (it is comprised of the n_best_per_sample positive answers along with the one no_answer)
        """
        # Initialize some variables
        top_candidates: List[QACandidate] = []
        n_candidates = sorted_candidates.shape[0]
        start_idx_candidates = set()
        end_idx_candidates = set()

        start_matrix_softmax_start = F.softmax(start_matrix[:, 0], axis=-1)
        end_matrix_softmax_end = F.softmax(end_matrix[0, :], axis=-1)

        # Iterate over all candidates and break when we have all our n_best candidates
        for candidate_idx in range(n_candidates):
            if len(top_candidates) == self.n_best_per_sample:
                break
            # Retrieve candidate's indices
            start_idx = sorted_candidates[candidate_idx, 0].item()
            end_idx = sorted_candidates[candidate_idx, 1].item()
            # Ignore no_answer scores which will be extracted later in this method
            if start_idx == 0 and end_idx == 0:
                continue
            if self.duplicate_filtering > -1 and (
                    start_idx in start_idx_candidates
                    or end_idx in end_idx_candidates):
                continue
            score = start_end_matrix[start_idx, end_idx].item()
            confidence = (start_matrix_softmax_start[start_idx].item() +
                          end_matrix_softmax_end[end_idx].item()) / 2
            top_candidates.append(
                QACandidate(
                    offset_answer_start=start_idx,
                    offset_answer_end=end_idx,
                    score=score,
                    answer_type="span",
                    offset_unit="token",
                    aggregation_level="passage",
                    passage_id=str(sample_idx),
                    confidence=confidence,
                ))
            if self.duplicate_filtering > -1:
                for i in range(0, self.duplicate_filtering + 1):
                    start_idx_candidates.add(start_idx + i)
                    start_idx_candidates.add(start_idx - i)
                    end_idx_candidates.add(end_idx + i)
                    end_idx_candidates.add(end_idx - i)

        no_answer_score = start_end_matrix[0, 0].item()
        no_answer_confidence = (start_matrix_softmax_start[0].item() +
                                end_matrix_softmax_end[0].item()) / 2
        top_candidates.append(
            QACandidate(
                offset_answer_start=0,
                offset_answer_end=0,
                score=no_answer_score,
                answer_type="no_answer",
                offset_unit="token",
                aggregation_level="passage",
                passage_id=None,
                confidence=no_answer_confidence,
            ))
        return top_candidates

    def formatted_preds_wrapper(self, logits: paddle.Tensor, **kwargs):
        """
        Format predictions for inference.

        :param logits: Model logits.
        :return: Predictions in the right format.
        """

        preds_final = []
        # This try catch is to deal with the fact that sometimes we collect preds before passing it to
        # formatted_preds (see Inferencer._get_predictions_and_aggregate()) and sometimes we don't
        # (see Inferencer._get_predictions())
        try:
            preds = kwargs["preds"]
            temp = preds
            preds_flat = [item for sublist in temp for item in sublist]
            kwargs["preds"] = preds_flat
        except KeyError:
            kwargs["preds"] = None

        logits_for_head = logits[0]
        preds = self.formatted_preds(logits=logits_for_head, **kwargs)
        # TODO This is very messy - we need better definition of what the output should look like
        if type(preds) == list:
            preds_final += preds
        elif type(preds) == dict and "predictions" in preds:
            preds_final.append(preds)

        return preds_final

    def formatted_preds(self,
                        preds: List[QACandidate],
                        baskets: List[SampleBasket],
                        logits: Optional[paddle.Tensor] = None,
                        **kwargs):
        """
        Takes a list of passage level predictions, each corresponding to one sample, and converts them into document level
        predictions. Leverages information in the SampleBaskets. Assumes that we are being passed predictions from
        ALL samples in the one SampleBasket i.e. all passages of a document. Logits should be None, because we have
        already converted the logits to predictions before calling formatted_preds.
        (see Inferencer._get_predictions_and_aggregate()).
        """
        # Unpack some useful variables
        # passage_start_t is the token index of the passage relative to the document (usually a multiple of doc_stride)
        # seq_2_start_t is the token index of the first token in passage relative to the input sequence (i.e. number of
        # special tokens and question tokens that come before the passage tokens)
        if logits or preds is None:
            logger.error(
                "QuestionAnsweringHead.formatted_preds() expects preds as input and logits to be None \
                            but was passed something different")

        samples = [s for b in baskets for s in b.samples]  # type: ignore
        ids = [s.id for s in samples]
        passage_start_t = [s.features[0]["passage_start_t"]
                           for s in samples]  # type: ignore
        seq_2_start_t = [s.features[0]["seq_2_start_t"]
                         for s in samples]  # type: ignore

        # Aggregate passage level predictions to create document level predictions.
        # This method assumes that all passages of each document are contained in preds
        # i.e. that there are no incomplete documents. The output of this step
        # are prediction spans
        preds_d = self.aggregate_preds(preds, passage_start_t, ids,
                                       seq_2_start_t)

        # Separate top_preds list from the no_ans_gap float.
        top_preds, no_ans_gaps = zip(*preds_d)

        # Takes document level prediction spans and returns string predictions
        doc_preds = self.to_qa_preds(top_preds, no_ans_gaps, baskets)

        return doc_preds

    def to_qa_preds(self, top_preds, no_ans_gaps, baskets):
        """
        Groups Span objects together in a QAPred object
        """
        ret = []

        # Iterate over each set of document level prediction
        for pred_d, no_ans_gap, basket in zip(top_preds, no_ans_gaps, baskets):

            # Unpack document offsets, clear text and id
            token_offsets = basket.raw["document_offsets"]
            pred_id = basket.id_external if basket.id_external else basket.id_internal

            # These options reflect the different input dicts that can be assigned to the basket
            # before any kind of normalization or preprocessing can happen
            question_names = ["question_text", "qas", "questions"]
            doc_names = ["document_text", "context", "text"]

            document_text = try_get(doc_names, basket.raw)
            question = self.get_question(question_names, basket.raw)
            ground_truth = self.get_ground_truth(basket)

            curr_doc_pred = QAPred(
                id=pred_id,
                prediction=pred_d,
                context=document_text,
                question=question,
                token_offsets=token_offsets,
                context_window_size=self.context_window_size,
                aggregation_level="document",
                ground_truth_answer=ground_truth,
                no_answer_gap=no_ans_gap,
            )
            ret.append(curr_doc_pred)
        return ret

    def aggregate_preds(self,
                        preds,
                        passage_start_t,
                        ids,
                        seq_2_start_t=None,
                        labels=None):
        """
        Aggregate passage level predictions to create document level predictions.
        This method assumes that all passages of each document are contained in preds
        i.e. that there are no incomplete documents. The output of this step
        are prediction spans. No answer is represented by a (-1, -1) span on the document level
        """
        # Initialize some variables
        n_samples = len(preds)
        all_basket_preds = {}
        all_basket_labels = {}

        # Iterate over the preds of each sample - remove final number which is the sample id and not needed for aggregation
        for sample_idx in range(n_samples):
            basket_id = ids[sample_idx]
            basket_id = basket_id.split("-")[:-1]
            basket_id = "-".join(basket_id)

            # curr_passage_start_t is the token offset of the current passage
            # It will always be a multiple of doc_stride
            curr_passage_start_t = passage_start_t[sample_idx]

            # This is to account for the fact that all model input sequences start with some special tokens
            # and also the question tokens before passage tokens.
            if seq_2_start_t:
                cur_seq_2_start_t = seq_2_start_t[sample_idx]
                curr_passage_start_t -= cur_seq_2_start_t

            # Converts the passage level predictions+labels to document level predictions+labels. Note
            # that on the passage level a no answer is (0,0) but at document level it is (-1,-1) since (0,0)
            # would refer to the first token of the document

            # pred1, pred2 = preds[sample_idx]
            pred_d = self.pred_to_doc_idxs(preds[sample_idx],
                                           curr_passage_start_t, sample_idx)
            if labels:
                label_d = self.label_to_doc_idxs(labels[sample_idx],
                                                 curr_passage_start_t)

            # Initialize the basket_id as a key in the all_basket_preds and all_basket_labels dictionaries
            if basket_id not in all_basket_preds:
                all_basket_preds[basket_id] = []
                all_basket_labels[basket_id] = []

            # Add predictions and labels to dictionary grouped by their basket_ids
            # passage-level -> document-level
            all_basket_preds[basket_id].append(pred_d)
            if labels:
                all_basket_labels[basket_id].append(label_d)

        # Pick n-best predictions and remove repeated labels
        idx = 0
        for k, v in all_basket_preds.items():
            pred1, pred2 = v[0]
            all_basket_preds[k] = self.reduce_preds(v)
            idx += 1
        # all_basket_preds = {k: self.reduce_preds(v) for k, v in all_basket_preds.items()}
        if labels:
            all_basket_labels = {
                k: self.reduce_labels(v)
                for k, v in all_basket_labels.items()
            }

        # Return aggregated predictions in order as a list of lists
        keys = [k for k in all_basket_preds]
        aggregated_preds = [all_basket_preds[k] for k in keys]
        if labels:
            labels = [all_basket_labels[k] for k in keys]
            return aggregated_preds, labels
        else:
            return aggregated_preds

    @staticmethod
    def pred_to_doc_idxs(pred, passage_start_t, sample_idx):
        """
        Converts the passage level predictions to document level predictions. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) qa_answer but will instead be represented by (-1, -1)
        """
        new_pred = []
        for qa_answer in pred:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            if start == 0:
                start = -1
            else:
                start += passage_start_t
                if start < 0:
                    logger.error("Start token index < 0 (document level)")
            if end == 0:
                end = -1
            else:
                end += passage_start_t
                if end < 0:
                    logger.error("End token index < 0 (document level)")
            qa_answer.to_doc_level(start, end)
            new_pred.append(qa_answer)
        return new_pred

    def reduce_preds(self, preds):
        """
        This function contains the logic for choosing the best answers from each passage. In the end, it
        returns the n_best predictions on the document level.
        """

        # Initialize variables
        passage_no_answer = []
        passage_best_score = []
        passage_best_confidence = []
        no_answer_scores = []
        no_answer_confidences = []
        n_samples = len(preds)

        # Iterate over the top predictions for each sample
        # Note: preds: [[QACandidate, QACandidate]]
        for sample_idx, sample_preds in enumerate(preds):
            best_pred = sample_preds[0]
            best_pred_score = best_pred.score
            best_pred_confidence = best_pred.confidence
            no_answer_score, no_answer_confidence = self.get_no_answer_score_and_confidence(
                sample_preds)
            no_answer_score += self.no_ans_boost
            # TODO we might want to apply some kind of a no_ans_boost to no_answer_confidence too
            no_answer = no_answer_score > best_pred_score
            passage_no_answer.append(no_answer)
            no_answer_scores.append(no_answer_score)
            no_answer_confidences.append(no_answer_confidence)
            passage_best_score.append(best_pred_score)
            passage_best_confidence.append(best_pred_confidence)

        # Get all predictions in flattened list and sort by score
        pos_answers_flat = []
        for sample_idx, passage_preds in enumerate(preds):
            for qa_candidate in passage_preds:
                # Todo(tianxin04): When all qa_candidate of preds has no answer, this func will occur error
                # Whether all qa_candidate has no answer is expected or not?
                if not (qa_candidate.offset_answer_start == -1
                        and qa_candidate.offset_answer_end == -1):
                    pos_answers_flat.append(
                        QACandidate(
                            offset_answer_start=qa_candidate.
                            offset_answer_start,
                            offset_answer_end=qa_candidate.offset_answer_end,
                            score=qa_candidate.score,
                            answer_type=qa_candidate.answer_type,
                            offset_unit="token",
                            aggregation_level="document",
                            passage_id=str(sample_idx),
                            n_passages_in_doc=n_samples,
                            confidence=qa_candidate.confidence,
                        ))

        # TODO add switch for more variation in answers, e.g. if varied_ans then never return overlapping answers
        pos_answer_dedup = self.deduplicate(pos_answers_flat)

        # This is how much no_ans_boost needs to change to turn a no_answer to a positive answer (or vice versa)
        no_ans_gap = -min([
            nas - pbs for nas, pbs in zip(no_answer_scores, passage_best_score)
        ])
        no_ans_gap_confidence = -min([
            nas - pbs
            for nas, pbs in zip(no_answer_confidences, passage_best_confidence)
        ])

        # "no answer" scores and positive answers scores are difficult to compare, because
        # + a positive answer score is related to a specific text qa_candidate
        # - a "no answer" score is related to all input texts
        # Thus we compute the "no answer" score relative to the best possible answer and adjust it by
        # the most significant difference between scores.
        # Most significant difference: change top prediction from "no answer" to answer (or vice versa)
        best_overall_positive_score = max(x.score for x in pos_answer_dedup)
        best_overall_positive_confidence = max(x.confidence
                                               for x in pos_answer_dedup)
        no_answer_pred = QACandidate(
            offset_answer_start=-1,
            offset_answer_end=-1,
            score=best_overall_positive_score - no_ans_gap,
            answer_type="no_answer",
            offset_unit="token",
            aggregation_level="document",
            passage_id=None,
            n_passages_in_doc=n_samples,
            confidence=best_overall_positive_confidence - no_ans_gap_confidence,
        )

        # Add no answer to positive answers, sort the order and return the n_best
        n_preds = [no_answer_pred] + pos_answer_dedup
        n_preds_sorted = sorted(
            n_preds,
            key=lambda x: x.confidence
            if self.use_confidence_scores_for_ranking else x.score,
            reverse=True)

        #n_best: The number of positive answer spans for each document.
        n_preds_reduced = n_preds_sorted[:self.n_best]
        return n_preds_reduced, no_ans_gap

    @staticmethod
    def get_no_answer_score_and_confidence(preds):
        for qa_answer in preds:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            score = qa_answer.score
            confidence = qa_answer.confidence
            if start == -1 and end == -1:
                return score, confidence
        raise Exception

    @staticmethod
    def deduplicate(flat_pos_answers):
        # Remove duplicate spans that might be twice predicted in two different passages
        seen = {}
        for qa_answer in flat_pos_answers:
            if (qa_answer.offset_answer_start,
                    qa_answer.offset_answer_end) not in seen:
                seen[(qa_answer.offset_answer_start,
                      qa_answer.offset_answer_end)] = qa_answer
            else:
                seen_score = seen[(qa_answer.offset_answer_start,
                                   qa_answer.offset_answer_end)].score
                if qa_answer.score > seen_score:
                    seen[(qa_answer.offset_answer_start,
                          qa_answer.offset_answer_end)] = qa_answer
        return list(seen.values())

    @staticmethod
    def get_question(question_names: List[str], raw_dict: Dict):
        # For NQ style dicts
        qa_name = None
        if "qas" in raw_dict:
            qa_name = "qas"
        elif "question" in raw_dict:
            qa_name = "question"
        if qa_name:
            if type(raw_dict[qa_name][0]) == dict:
                return raw_dict[qa_name][0]["question"]
        return try_get(question_names, raw_dict)

    @staticmethod
    def get_ground_truth(basket: SampleBasket):
        if "answers" in basket.raw:
            return basket.raw["answers"]
        elif "annotations" in basket.raw:
            return basket.raw["annotations"]
        else:
            return None
