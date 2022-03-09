from paddlenlp.transformers.ernie.modeling import ErnieLMPredictionHead, ErniePretrainedModel
import paddle.nn as nn
import paddle
class ErnieForMLMPretraining(ErniePretrainedModel):
    def __init__(self, ernie):
        super(ErnieForMLMPretraining, self).__init__()
        self.ernie = ernie
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            mean=0.0, std=self.ernie.initializer_range))
        self.cls = ErnieLMPredictionHead(
            self.ernie.config["hidden_size"],
            self.ernie.config["vocab_size"],
            self.ernie.config["hidden_act"],
            # embedding_weights=self.ernie.embeddings.word_embeddings.weight,
            weight_attr=weight_attr, )

        self.apply(self.init_weights)

    def forward(self,
            input_ids,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None):           
        with paddle.static.amp.fp16_guard():
            outputs = self.ernie(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask)
            sequence_output, pooled_output = outputs[:2]
            prediction_scores = self.cls(
                sequence_output)
            return prediction_scores
