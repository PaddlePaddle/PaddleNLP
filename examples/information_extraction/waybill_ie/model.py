import paddle.nn as nn
from paddlenlp.transformers import ErniePretrainedModel
from paddlenlp.layers.crf import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss


class ErnieCrfForTokenClassification(nn.Layer):
    def __init__(self, ernie, crf_lr=100):
        super().__init__()
        self.num_classes = ernie.num_classes
        self.ernie = ernie  # allow ernie to be config
        self.crf = LinearChainCrf(
            self.num_classes,  #crf_lr,
            with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(
            self.crf.transitions, with_start_stop_tag=False)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                lengths=None,
                labels=None):
        logits = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids)

        if labels is not None:
            loss = self.crf_loss(logits, lengths, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(logits, lengths)
            return prediction


# class ErnieCrfForTokenClassification(ErniePretrainedModel):
#     def __init__(self, ernie, num_classes, crf_lr=0.01, dropout=None):
#         super().__init__()
#         self.num_classes = num_classes
#         self.ernie = ernie  # allow ernie to be config
#         self.dropout = nn.Dropout(dropout if dropout is not None else
#                                   self.ernie.config["hidden_dropout_prob"])
#         self.classifier = nn.Linear(self.ernie.config["hidden_size"],
#                                     num_classes)
#         self.crf = LinearChainCrf(self.num_classes, crf_lr,
#                                   with_start_stop_tag=False)
#         self.crf_loss = LinearChainCrfLoss(self.crf)
#         self.viterbi_decoder = ViterbiDecoder(self.crf.transitions,
#                                               with_start_stop_tag=False)
#         self.apply(self.init_weights)

#     def forward(self,
#                 input_ids,
#                 token_type_ids=None,
#                 position_ids=None,
#                 attention_mask=None,
#                 lengths=None,
#                 labels=None):
#         sequence_output, _ = self.ernie(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,)

#         # sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)

#         if labels is not None:
#             loss = self.crf_loss(logits, lengths, labels)
#             return loss
#         else:
#             _, prediction = self.viterbi_decoder(logits, lengths)
#             return prediction
