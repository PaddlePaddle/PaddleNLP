# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import copy
import os
import itertools

import LAC
import numpy as np
import paddle
import paddle.nn.functional as F
from ..data import Vocab, Pad
from .utils import download_file, static_mode_guard, dygraph_mode_guard
from .task import Task
from .models import BiAffineParser

URLS = {"ddparser": 
            ["http://10.21.226.184:8072/ddparser.tar.gz", None], 
        "ddparser-ernie-1.0":
            ["http://10.21.226.184:8072/ddparser-ernie-1.0.tar.gz", None],  
        "ddparser-ernie-gram-zh":
            ["http://10.21.226.184:8072/ddparser-ernie-gram-zh.tar.gz", None],
        }

usage = r"""
           from paddlenlp.taskflow import TaskFlow 

           ddp = TaskFlow("dependency_parsing")
           ddp("百度是一家高科技公司")
           '''
           [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': ['2', '0', '5', '5', '2'], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
           '''
           ddp(["百度是一家高科技公司", "他送了一本书"])
           '''
           [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': ['2', '0', '5', '5', '2'], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': ['2', '0', '2', '5', '2'], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]
           '''

           ddp = TaskFlow("dependency_parsing", prob=True, use_pos=True)
           ddp("百度是一家高科技公司")
           '''
           [{'word': ['百度', '是', '一家', '高科技', '公司'], 'postag': ['ORG', 'v', 'm', 'n', 'n'], 'head': ['2', '0', '5', '5', '2'], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB'], 'prob': [1.0, 1.0, 1.0, 1.0, 1.0]}]
           '''

           ddp-ernie-1.0 = TaskFlow("dependency_parsing", encoding_model="ernie-1.0")
           ddp-ernie-1.0("百度是一家高科技公司")
           '''
           [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': ['2', '0', '5', '5', '2'], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
           '''
           ddp-ernie-1.0(["百度是一家高科技公司", "他送了一本书"])
           '''
           [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': ['2', '0', '5', '5', '2'], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': ['2', '0', '2', '5', '2'], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]
           '''

         """


class DDParserTask(Task):
    """
    DDParser task to analyze the dependency relationship between words in a sentence 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        static_mode(bool): The flag to control in the static/dygraph mode.
        encoding_model(string): The word encoder for ddparser.
        tree(bool): Ensure the output conforms to the tree structure.
        prob(bool): Whether to return the probability of predicted heads.
        use_pos(bool): Whether to return the postag.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, 
                 task, 
                 model, 
                 static_mode, 
                 encoding_model="lstm-pe", 
                 tree=True,
                 prob=False,
                 use_pos=False,
                 batch_size=1,
                 **kwargs):
        super().__init__(
            task=task, model=model, static_mode=static_mode, **kwargs)
        self._usage = usage
        self.encoding_model = encoding_model

        if self.encoding_model == "ernie-gram-zh":
            self.model_dir = "ddparser-ernie-gram-zh"
        elif self.encoding_model == "ernie-1.0":
            self.model_dir = "ddparser-ernie-1.0"
        elif self.encoding_model == "lstm-pe":
            self.model_dir = "ddparser"
        else:
            raise ValueError("The encoding model should be lstm-pe, ernie-1.0 or ernie-gram-zh")

        word_vocab_path = download_file(
            self._task_path, self.model_dir + os.path.sep + "word_vocab.json",
            URLS[self.model_dir][0], URLS[self.model_dir][1])
        rel_vocab_path = download_file(
            self._task_path, self.model_dir + os.path.sep + "rel_vocab.json",
            URLS[self.model_dir][0], URLS[self.model_dir][1])
        self.word_vocab = Vocab.from_json(word_vocab_path)
        self.rel_vocab = Vocab.from_json(rel_vocab_path)
        self.word_pad_index = self.word_vocab.to_indices("[PAD]")
        self.word_bos_index = self.word_vocab.to_indices("[CLS]")
        self.word_eos_index = self.word_vocab.to_indices("[SEP]")
        self.tree = tree
        self.prob = prob
        self.use_pos = use_pos
        self.batch_size = batch_size

        self.lac = LAC.LAC(mode="lac" if self.use_pos else "seg", use_cuda=True)
        if self.static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)

    def _construct_input_spec(self):
        """
       Construct the input spec for the convert dygraph model to static model.
       """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),                         
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = BiAffineParser(
            encoding_model=self.encoding_model,
            n_rels=len(self.rel_vocab),
            n_words=len(self.word_vocab),
            pad_index=self.word_pad_index,
            eos_index=self.word_eos_index,
        )
        # Load the model parameter for the predict
        state_dict = paddle.load(
            os.path.join(self._task_path, self.model_dir, "model.pdparams"))
        model_instance.set_dict(state_dict)
        model_instance.eval()
        self._model = model_instance

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        return None

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = inputs[0]
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, str) and not isinstance(inputs, list):
            raise TypeError(
                "Invalid inputs, input text should be str or list of str, {type(inputs)} found!"
            )
        # Get the config from the kwargs
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False    

        lac_results = []
        position = 0

        while position < len(inputs):
            lac_results += self.lac.run(inputs[position:position + self.batch_size])
            position += self.batch_size   

        outputs = {}
        if not self.use_pos:
            outputs['words'] = lac_results
        else:
            outputs['words'], outputs['postags'] = [raw for raw in zip(*lac_results)]

        examples = []
        for text in outputs['words']:
            example = {
                "FORM": text, 
            }
            example = convert_example(
                example,
                vocabs=[self.word_vocab, self.rel_vocab],
            )
            examples.append(example)

        batches = [
            examples[idx:idx + self.batch_size]
            for idx in range(0, len(examples), self.batch_size)
        ]  

        def batchify_fn(batch):
            raw_batch = [raw for raw in zip(*batch)]
            batch = [pad_sequence(data) for data in raw_batch]
            return batch
        
        batches = [flat_words(batchify_fn(batch)[0]) for batch in batches]

        outputs['data_loader'] = batches
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """

        arcs, rels, probs = [], [], []
        if not self.static_mode:
            with dygraph_mode_guard():
                for batch in inputs['data_loader']:
                    words, wp = batch
                    words = paddle.to_tensor(words)
                    wp = paddle.to_tensor(wp)

                    s_arc, s_rel, words = self._model(words, wp)
                    
                    words = words.numpy()
                    mask = np.logical_and(
                        np.logical_and(words != self.word_pad_index, words != self.word_bos_index),
                        words != self.word_eos_index, 
                    )
                    arc_preds, rel_preds = decode(s_arc, s_rel, mask, self.tree)

                    arcs.extend([arc_pred[m] for arc_pred, m in zip(arc_preds, mask)])
                    rels.extend([rel_pred[m] for rel_pred, m in zip(rel_preds, mask)])  

                    if self.prob:
                        arc_probs = probability(s_arc.numpy(), arc_preds)   
                        probs.extend([arc_prob[m] for arc_prob, m in zip(arc_probs, mask)])                
        else:
            with static_mode_guard():
                for batch in inputs['data_loader']:
                    data_dict = {}
                    for name, value in zip(self._static_feed_names, batch):
                        data_dict[name] = value

                    s_arc, s_rel, words = self._exe.run(
                        self._static_program,
                        feed=data_dict,
                        fetch_list=self._static_fetch_targets)

                    mask = np.logical_and(
                        np.logical_and(words != self.word_pad_index, words != self.word_bos_index),
                        words != self.word_eos_index, 
                    )
                    arc_preds, rel_preds = decode(s_arc, s_rel, mask, self.tree)

                    arcs.extend([arc_pred[m] for arc_pred, m in zip(arc_preds, mask)])
                    rels.extend([rel_pred[m] for rel_pred, m in zip(rel_preds, mask)])
                    if self.prob:
                        arc_probs = probability(s_arc, arc_preds)  
                        probs.extend([arc_prob[m] for arc_prob, m in zip(arc_probs, mask)])     
        inputs['arcs'] = arcs
        inputs['rels'] = rels
        inputs['probs'] = probs
        return inputs

    def _postprocess(self, inputs):

        arcs = inputs['arcs']
        rels = inputs['rels']
        words = inputs['words']
        arcs = [[str(s) for s in seq] for seq in arcs]
        rels = [self.rel_vocab.to_tokens(seq) for seq in rels]   

        results = []

        for word, arc, rel in zip(words, arcs, rels):
            single_result = {
                'word': word,
                'head': arc,
                'deprel': rel,
            }
            results.append(single_result)   

        if self.use_pos:
            postags = inputs['postags']
            for single_result, postag in zip(results, postags):
                single_result['postag'] = postag  

        if self.prob:
            probs = inputs['probs']
            probs = [[round(p, 3) for p in seq.tolist()] for seq in probs]
            for single_result, prob in zip(results, probs):
                single_result['prob'] = prob

        return results


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Fill sequences(np.ndarray) into a fixed-length matrix."""
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        assert fix_len >= max_len, "fix_len is too small."
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = np.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


def convert_example(example,
                    vocabs, 
                    fix_len=20):
    word_vocab, rel_vocab = vocabs

    word_bos_index = word_vocab.to_indices("[CLS]")
    word_eos_index = word_vocab.to_indices("[SEP]") 

    words = []
    for word in example["FORM"]:
        words.append([word_vocab.to_indices(char) for char in word])

    words = [[word_bos_index]] + words + [[word_eos_index]]
    return [
        pad_sequence([np.array(ids[:fix_len], dtype=int) 
        for ids in words], fix_len=fix_len)
    ]


def flat_words(words, pad_index=0):
    mask = words != pad_index
    lens = np.sum(mask.astype(int), axis=-1)
    position = np.cumsum(lens + (lens == 0).astype(int), axis=1) - 1
    lens = np.sum(lens, -1)
    words = words.ravel()[np.flatnonzero(words)]

    sequences = []
    idx = 0
    for l in lens:
        sequences.append(words[idx:idx+l])
        idx += l
    words = Pad(pad_val=pad_index)(sequences)

    max_len = words.shape[1]

    mask = (position >= max_len).astype(int)
    position = position * np.logical_not(mask) + mask * (max_len - 1)
    return words, position


def probability(s_arc, arc_preds):
    s_arc = s_arc - s_arc.max(axis=-1).reshape(list(s_arc.shape)[:-1]+[1])
    s_arc = np.exp(s_arc) / np.exp(s_arc).sum(axis=-1).reshape(list(s_arc.shape)[:-1]+[1])

    arc_probs = [
        s[np.arange(len(arc_pred)), arc_pred]
        for s, arc_pred in zip(s_arc, arc_preds)
    ]
    return arc_probs


def decode(s_arc, s_rel, mask, tree=True):

    lens = np.sum(mask.astype(int), axis=-1)
    arc_preds = np.argmax(s_arc, axis=-1)

    bad = [not istree(seq[:i + 1]) for i, seq in zip(lens, arc_preds)]
    if tree and any(bad):
        arc_preds[bad] = eisner(s_arc[bad], mask[bad])
    
    rel_preds = np.argmax(s_rel, axis=-1)
    rel_preds = [
        rel_pred[np.arange(len(arc_pred)), arc_pred]
        for arc_pred, rel_pred in zip(arc_preds, rel_preds)
    ]
    return arc_preds, rel_preds


def eisner(scores, mask):
    """Eisner algorithm is a general dynamic programming decoding algorithm for bilexical grammar.

    Args：
        scores: Adjacency matrix，shape=(batch, seq_len, seq_len)
        mask: mask matrix，shape=(batch, sql_len)

    Returns:
        output，shape=(batch, seq_len)，the index of the parent node corresponding to the token in the query

    """
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.transpose(2, 1, 0)
    # score for incomplete span
    s_i = np.full_like(scores, float('-inf'))
    # score for complete span
    s_c = np.full_like(scores, float('-inf'))
    # incompelte span position for backtrack
    p_i = np.zeros((seq_len, seq_len, batch_size), dtype=np.int64)
    # compelte span position for backtrack
    p_c = np.zeros((seq_len, seq_len, batch_size), dtype=np.int64)
    # set 0 to s_c.diagonal
    s_c = fill_diagonal(s_c, 0)
    # contiguous
    s_c = np.ascontiguousarray(s_c)
    s_i = np.ascontiguousarray(s_i)
    for w in range(1, seq_len):
        n = seq_len - w
        starts = np.arange(n, dtype=np.int64)[np.newaxis, :]
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        ilr = ilr.transpose(2, 0, 1)
        # scores.diagonal(-w).shape:[batch, n]
        il = ilr + scores.diagonal(-w)[..., np.newaxis]
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1), il.argmax(-1)
        s_i = fill_diagonal(s_i, il_span, offset=-w)
        p_i = fill_diagonal(p_i, il_path + starts, offset=-w)

        ir = ilr + scores.diagonal(w)[..., np.newaxis]
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1), ir.argmax(-1)
        s_i = fill_diagonal(s_i, ir_span, offset=w)
        p_i = fill_diagonal(p_i, ir_path + starts, offset=w)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl = cl.transpose(2, 0, 1)
        cl_span, cl_path = cl.max(-1), cl.argmax(-1)
        s_c = fill_diagonal(s_c, cl_span, offset=-w)
        p_c = fill_diagonal(p_c, cl_path + starts, offset=-w)

        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr = cr.transpose(2, 0, 1)
        cr_span, cr_path = cr.max(-1), cr.argmax(-1)
        s_c = fill_diagonal(s_c, cr_span, offset=w)
        s_c[0, w][np.not_equal(lens, w)] = float('-inf')
        p_c = fill_diagonal(p_c, cr_path + starts + 1, offset=w)

    predicts = []
    p_c = p_c.transpose(2, 0, 1)
    p_i = p_i.transpose(2, 0, 1)
    for i, length in enumerate(lens.tolist()):
        heads = np.ones(length + 1, dtype=np.int64)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads)

    return pad_sequence(predicts, fix_len=seq_len)


class NODE:
    """NODE class"""
    def __init__(self, id=None, parent=None):
        self.lefts = []
        self.rights = []
        self.id = int(id)
        self.parent = parent if parent is None else int(parent)


class DepTree:
    """
    DepTree class, used to check whether the prediction result is a project Tree.
    A projective tree means that you can project the tree without crossing arcs.
    """
    def __init__(self, sentence):
        # set root head to -1
        sentence = copy.deepcopy(sentence)
        sentence[0] = -1
        self.sentence = sentence
        self.build_tree()
        self.visit = [False] * len(sentence)

    def build_tree(self):
        """Build the tree"""
        self.nodes = [NODE(index, p_index) for index, p_index in enumerate(self.sentence)]
        # set root
        self.root = self.nodes[0]
        for node in self.nodes[1:]:
            self.add(self.nodes[node.parent], node)

    def add(self, parent, child):
        """Add a child node"""
        if parent.id is None or child.id is None:
            raise Exception("id is None")
        if parent.id < child.id:
            parent.rights = sorted(parent.rights + [child.id])
        else:
            parent.lefts = sorted(parent.lefts + [child.id])

    def judge_legal(self):
        """Determine whether it is a project tree"""
        target_seq = list(range(len(self.nodes)))
        if len(self.root.lefts + self.root.rights) != 1:
            return False
        cur_seq = self.inorder_traversal(self.root)
        if target_seq != cur_seq:
            return False
        else:
            return True

    def inorder_traversal(self, node):
        """Inorder traversal"""
        if self.visit[node.id]:
            return []
        self.visit[node.id] = True
        lf_list = []
        rf_list = []
        for ln in node.lefts:
            lf_list += self.inorder_traversal(self.nodes[ln])
        for rn in node.rights:
            rf_list += self.inorder_traversal(self.nodes[rn])

        return lf_list + [node.id] + rf_list


def istree(sequence):
    """Is the sequence a project tree"""
    return DepTree(sequence).judge_legal()
