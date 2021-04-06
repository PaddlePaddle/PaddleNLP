import json

import paddle
import paddle.nn as nn
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Pad, Tuple
from paddlenlp.transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer


class WordtagPredictor(object):
    """Predictor of wordtag model.
    """

    def __init__(self, model_dir, tag_path, linking_path=None):
        """Initialize method of the predictor.

        Args:
            model_dir: The pre-trained model checkpoint dir.
            tag_path: The tag vocab path.
            linking_path:if you want to use linking mode, you should load link feature using.
        """
        self._tags_to_index, self._index_to_tags = self._load_labels(tag_path)

        self._model = ErnieCtmWordtagModel.from_pretrained(
            model_dir,
            num_cls_label=4,
            num_tag=len(self._tags_to_index),
            ignore_index=self._tags_to_index["O"])
        self._model.eval()

        self._tokenizer = ErnieCtmTokenizer.from_pretrained(model_dir)
        self._summary_num = self._model.ernie_ctm.content_summary_index + 1
        self.linking = False
        if linking_path is not None:
            self.linking_dict = {}
            with open(linking_path, encoding="utf-8") as fp:
                for line in fp:
                    data = json.loads(line)
                    if data["label"] not in self.linking_dict:
                        self.linking_dict[data["label"]] = []
                    self.linking_dict[data["label"]].append({
                        "sid": data["sid"],
                        "cls": paddle.to_tensor(data["cls1"]).unsqueeze(0),
                        "term": paddle.to_tensor(data["term"]).unsqueeze(0)
                    })
            self.linking = True
            self.sim_fct = nn.CosineSimilarity(dim=1)

    @property
    def summary_num(self):
        """Number of model summary token
        """
        return self._summary_num

    @staticmethod
    def _load_labels(tag_path):
        tags_to_idx = {}
        i = 0
        with open(tag_path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                tags_to_idx[line] = i
                i += 1
        idx_to_tags = dict(zip(*(tags_to_idx.values(), tags_to_idx.keys())))
        return tags_to_idx, idx_to_tags

    def _pre_process_text(self, input_texts, max_seq_len=128, batch_size=1):
        infer_data = []
        max_length = 0
        for text in input_texts:
            tokens = ["[CLS%i]" % i
                      for i in range(1, self.summary_num)] + list(text)
            tokenized_input = self._tokenizer(
                tokens,
                return_length=True,
                is_split_into_words=True,
                max_seq_len=max_seq_len)
            infer_data.append([
                tokenized_input['input_ids'], tokenized_input['token_type_ids'],
                tokenized_input['seq_len']
            ])
        infer_ds = MapDataset(infer_data)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id),  # token_type_ids
            Stack(),  # seq_len
        ): fn(samples)

        infer_data_loader = paddle.io.DataLoader(
            infer_ds,
            collate_fn=batchify_fn,
            num_workers=0,
            batch_size=batch_size,
            shuffle=False,
            return_list=True)

        return infer_data_loader

    def _decode(self, batch_texts, batch_pred_tags):
        batch_results = []
        for i, pred_tags in enumerate(batch_pred_tags):
            pred_words, pred_word = [], []
            text = batch_texts[i]
            for j, tag in enumerate(pred_tags[self.summary_num:-1]):
                if j > len(text) + self.summary_num - 1:
                    continue
                pred_label = self._index_to_tags[tag]
                if pred_label.find("-") != -1:
                    _, label = pred_label.split("-")
                else:
                    label = pred_label
                if pred_label.startswith("S") or pred_label.startswith("O"):
                    pred_words.append({
                        "item": text[j],
                        "offset": 0,
                        "wordtag_label": label
                    })
                else:
                    pred_word.append(text[j])
                    if pred_label.startswith("E"):
                        pred_words.append({
                            "item": "".join(pred_word),
                            "offset": 0,
                            "wordtag_label": label
                        })
                        del pred_word[:]
            for i in range(len(pred_words)):
                if i > 0:
                    pred_words[i]["offset"] = pred_words[i - 1]["offset"] + len(
                        pred_words[i - 1]["item"])
                pred_words[i]["length"] = len(pred_words[i]["item"])
            result = {"text": text, "items": pred_words}
            batch_results.append(result)
        return batch_results

    @paddle.no_grad()
    def run(self,
            input_texts,
            max_seq_len=128,
            batch_size=1,
            return_hidden_states=None):
        """Predict a input text by wordtag.

        Args:
            input_text: input text.
            max_seq_len: max sequence length.
            batch_size: Batch size per GPU/CPU for training.

        Returns:
            dict -- wordtag results.
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        if not isinstance(input_texts, str) and not isinstance(input_texts,
                                                               list):
            raise TypeError(
                f"Bad inputs, input text should be str or list of str, {type(input_texts)} found!"
            )
        infer_data_loader = self._pre_process_text(input_texts, max_seq_len,
                                                   batch_size)
        all_pred_tags = []
        with paddle.no_grad():
            for batch in infer_data_loader:
                input_ids, token_type_ids, seq_len = batch
                seq_logits, cls_logits = self._model(
                    input_ids, token_type_ids, lengths=seq_len)
                scores, pred_tags = self._model.viterbi_decoder(seq_logits,
                                                                seq_len)
                all_pred_tags += pred_tags.numpy().tolist()

        results = self._decode(input_texts, all_pred_tags)
        outputs = results
        if return_hidden_states is True:
            outputs = (results, ) + (seq_logits, cls_logits)
        return outputs

    def _post_linking(self, pred_res, hidden_states):
        for pred in pred_res:
            for item in pred["items"]:
                if item["item"] in self.linking_dict:
                    item_vectors = self.linking_dict[item["item"]]
                    item_pred_vector = hidden_states[1]

                    res = []
                    for item_vector in item_vectors:
                        vec = item_vector["cls"]
                        similarity = self.sim_fct(vec, item_pred_vector)
                        res.append({
                            "sid": item_vector["sid"],
                            "cosine": similarity.item()
                        })
                    res.sort(key=lambda d: -d["cosine"])
                    item["link"] = res

    def run_with_link(self, input_text):
        """Predict wordtag results with term linking.

        Args:
            input_text: input text

        Raises:
            ValueError: raise ValueError if is not linking mode.

        Returns:
            pred_res: result with linking.
        """
        if self.linking is False:
            raise ValueError(
                "Not linking mode, you should initialize object by ``WordtagPredictor(model_dir, linking_path)``."
            )
        pred_res = self.run(input_text, return_hidden_states=True)
        self._post_linking(pred_res[0], pred_res[1:])
        return pred_res[0]
