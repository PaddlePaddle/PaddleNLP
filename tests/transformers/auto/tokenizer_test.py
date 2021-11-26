from paddlenlp.transformers import AutoTokenizer
from collections import OrderedDict
from paddlenlp.transformers import *


def from_built_in_models():
    print('From_built_in_models:------------------')
    # From built-in pretrained models
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/built_in_bert_auto'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/built_in_bert')

    tokenizer = AutoTokenizer.from_pretrained('plato-mini')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/built_in_ut_auto')
    tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/built_in_ut_auto')

    tokenizer = AutoTokenizer.from_pretrained('bigbird-base-uncased')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/built_in_bigbird_auto'
    )
    tokenizer = BigBirdTokenizer.from_pretrained('bigbird-base-uncased')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/built_in_bigbird')


def from_local_dir():
    print('From_local_dir:--------------------------')
    # From local dir path
    tokenizer = AutoTokenizer.from_pretrained(
        ('/Users/huhuiwen01/notebook/saved_model/my_bert'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer = AutoTokenizer.from_pretrained(
        ('/Users/huhuiwen01/notebook/saved_model/my_bart'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer = AutoTokenizer.from_pretrained(
        ('/Users/huhuiwen01/notebook/saved_model/my_bigbird'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))


def from_community_models():
    print('From_community_models:-------------------')
    #From community-contributed pretrained models
    tokenizer = AutoTokenizer.from_pretrained(
        'yingyibiao/bert-base-uncased-sst-2-finetuned')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/community_bert_auto'
    )
    tokenizer = BertTokenizer.from_pretrained(
        'yingyibiao/bert-base-uncased-sst-2-finetuned')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/community_bert')

    # community 无 init_class
    tokenizer = AutoTokenizer.from_pretrained(
        'junnyu/ckiplab-bert-base-chinese-ner')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/community_junnyu_bert_auto'
    )
    tokenizer = BertTokenizer.from_pretrained(
        'junnyu/ckiplab-bert-base-chinese-ner')
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer.save_pretrained(
        '/Users/huhuiwen01/Downloads/huhuiwen/saved_tokenizer/community_junnyu_bert'
    )


if __name__ == '__main__':

    from_built_in_models()
    from_local_dir()
    from_community_models()
