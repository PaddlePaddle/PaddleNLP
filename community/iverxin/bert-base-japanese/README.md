# BERT base Japanese (IPA dictionary)

This is a [BERT](https://github.com/google-research/bert) model pretrained on texts in the Japanese language.

This version of the model processes input texts with word-level tokenization based on the IPA dictionary, followed by the WordPiece subword tokenization.

The codes for the pretraining are available at [cl-tohoku/bert-japanese](https://github.com/cl-tohoku/bert-japanese/tree/v1.0).

## Model architecture

The model architecture is the same as the original BERT base model; 12 layers, 768 dimensions of hidden states, and 12 attention heads.

## Training Data

The model is trained on Japanese Wikipedia as of September 1, 2019.

To generate the training corpus, [WikiExtractor](https://github.com/attardi/wikiextractor) is used to extract plain texts from a dump file of Wikipedia articles.

The text files used for the training are 2.6GB in size, consisting of approximately 17M sentences.

## Tokenization

The texts are first tokenized by [MeCab](https://taku910.github.io/mecab/) morphological parser with the IPA dictionary and then split into subwords by the WordPiece algorithm.

The vocabulary size is 32000.

## Training

The model is trained with the same configuration as the original BERT; 512 tokens per instance, 256 instances per batch, and 1M training steps.

## Licenses

The pretrained models are distributed under the terms of the [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

## Acknowledgments

For training models, we used Cloud TPUs provided by [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc/) program.


## Usage
```python
import paddle
from paddlenlp.transformers import BertJapaneseTokenizer, BertForMaskedLM

path = "iverxin/bert-base-japanese/"
tokenizer = BertJapaneseTokenizer.from_pretrained(path)
model = BertForMaskedLM.from_pretrained(path)
text1 = "こんにちは"

model.eval()
inputs = tokenizer(text1)
inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
output = model(**inputs)
print(output.shape)
```


## Weights source
https://huggingface.co/cl-tohoku/bert-base-japanese
