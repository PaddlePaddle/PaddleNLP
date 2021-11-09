from paddlenlp.transformers import AutoTokenizer


def from_local_dir():
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


def from_built_in_models():
    # From built-in pretrained models
    tokenizer = AutoTokenizer.from_pretrained(('bert-base-cased'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer = AutoTokenizer.from_pretrained(('plato-mini'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))
    tokenizer = AutoTokenizer.from_pretrained(('bigbird-base-uncased'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))


def from_community_models():
    # From community-contributed pretrained models
    tokenizer = AutoTokenizer.from_pretrained(
        ('yingyibiao/bert-base-uncased-sst-2-finetuned'))
    print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))

    #tokenizer = AutoTokenizer.from_pretrained(('junnyu/ckiplab-bert-base-chinese-ner'))
    #print(tokenizer("Welcome to use PaddlePaddle and PaddleNLP!"))


if __name__ == '__main__':
    from_local_dir()
    from_built_in_models()
    from_community_models()
