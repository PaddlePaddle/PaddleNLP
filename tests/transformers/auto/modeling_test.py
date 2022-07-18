import paddle
from paddlenlp.transformers.auto.modeling import *
from paddlenlp.transformers import *
import warnings


def from_built_in_model():
    print('From_built_in_models:-------------------------------')

    # model test
    model = AutoModel.from_pretrained('bert-base-uncased')
    print(type(model))
    #model = AutoModel.from_pretrained('unimo-text-1.0')
    #model = AutoModel.from_pretrained('plato-mini')
    #model = AutoModel.from_pretrained('unified_transformer-12L-cn')

    # pretraining test
    #model = AutoModelForPretraining.from_pretrained('roformer-chinese-small')

    model = AutoModel.from_pretrained('roformer-chinese-small',
                                      task='ForPretraining')
    print(type(model))

    #model = AutoModelForPretraining.from_pretrained('tinybert-4l-312d')

    # lm_head test
    #model = AutoModelWithLMHead.from_pretrained('gpt-cpm-large-cn')
    #print(type(model))
    #model = AutoModelWithLMHead.from_pretrained('unified_transformer-12L-cn')
    #model = AutoModelWithLMHead.from_pretrained('unimo-text-1.0')

    # masked lm test

    model = AutoModelForMaskedLM.from_pretrained('albert-base-v1')
    print(type(model))
    #model = AutoModelForMaskedLM.from_pretrained('bart-base')
    #model = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased')
    #model = AutoModelForMaskedLM.from_pretrained('mpnet-base')

    # sequence_classification test
    model = AutoModelForSequenceClassification.from_pretrained('rbt3')
    print(type(model))
    #model = AutoModelForSequenceClassification.from_pretrained('roberta-wwm-ext')
    #model = AutoModelForSequenceClassification.from_pretrained('roformer-chinese-small')

    # multiple choice test
    model = AutoModelForMultipleChoice.from_pretrained('albert-base-v1')
    print(type(model))
    #model = AutoModelForMultipleChoice.from_pretrained('nezha-base-chinese')

    # QA test
    model = AutoModelForQuestionAnswering.from_pretrained('nezha-base-chinese')
    print(type(model))
    #model = AutoModelForQuestionAnswering.from_pretrained('ernie-3.0-medium-zh')
    #model = AutoModelForQuestionAnswering.from_pretrained('ernie-gram-zh')

    # token_classification test
    model = AutoModelForTokenClassification.from_pretrained('electra-small',
                                                            num_classes=2)
    print(type(model))
    #model = AutoModelForTokenClassification.from_pretrained('rbt3')
    #model = AutoModelForTokenClassification.from_pretrained('skep_ernie_1.0_large_ch')
    #model = AutoModelForTokenClassification.from_pretrained('plato-mini')

    # encoder, decoder test
    model = AutoDecoder.from_pretrained("bart-base", vocab_size=20000)
    print(type(model))
    model = AutoEncoder.from_pretrained("bart-base", vocab_size=20000)
    print(type(model))
    #model = AutoEncoder.from_pretrained("bart-base", vocab_size = 20000)

    # discriminator, generotor test
    model = AutoGenerator.from_pretrained("convbert-base")
    print(type(model))
    model = AutoDiscriminator.from_pretrained("convbert-base")
    print(type(model))
    #model = AutoModelForPretraining.from_pretrained('roberta-wwm-ext')
    #model = AutoGenerator.from_pretrained("electra-small")
    #model = AutoDiscriminator.from_pretrained("convbert-base")
    #model = AutoDiscriminator.from_pretrained("electra-small")

    # CausalLM test
    model = AutoModelForCausalLM.from_pretrained('blenderbot-3B')
    print(type(model))

    model = AutoModelForConditionalGeneration.from_pretrained(
        'blenderbot_small-90M')
    print(type(model))

    model = AutoModelForCausalLM.from_pretrained('blenderbot_small-90M')
    print(type(model))


def from_local_dir():
    print('From_local_dir:-----------------------------------------')
    model = AutoModel.from_pretrained(('saved_model/my_bert_model'))
    print(type(model))
    model = AutoModelForSequenceClassification.from_pretrained(
        ('saved_model/my_bert_model_for_pretraining'))
    print(type(model))


def from_community_model():
    print('From_community_models:---------------------------------')
    #model = AutoModelForSequenceClassification.from_pretrained(
    #   'yingyibiao/bert-base-uncased-sst-2-finetuned')
    #print(type(model))
    model = AutoModelForSequenceClassification.from_pretrained(
        'junnyu/ckiplab-bert-base-chinese-ner')
    print(type(model))


if __name__ == '__main__':

    from_built_in_model()
    from_local_dir()
    from_community_model()
