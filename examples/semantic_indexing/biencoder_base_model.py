import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class BiEncoder(nn.Layer):
    def __init__(self,question_encoder,context_encoder,dropout,output_emb_size,state=None):
        super(BiEncoder, self).__init__()
        self.state = state
        if self.state == None:
            self.question_encoder = question_encoder
            self.context_encoder = context_encoder
        elif self.state == "FORQUESTION":
            self.question_encoder = question_encoder
        elif self.state == "FORCONTEXT":
            self.context_encoder = context_encoder
        self.dropout = self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
        self.emb_reduce_linear = paddle.nn.Linear(
            768, output_emb_size, weight_attr=weight_attr)

    def get_question_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None):

        _, cls_embedding = self.question_encoder(input_ids, token_type_ids, position_ids,attention_mask)

        """cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)"""

        return cls_embedding

    def get_context_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None):

        _, cls_embedding = self.context_encoder(input_ids, token_type_ids, position_ids,attention_mask)

        """cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)"""

        return cls_embedding

    def forward(self,
                question_id,
                question_segments,
                question_attn_mask,
                context_ids,
                context_segments,
                context_attn_mask,
                    ):

        question_pooled_out = self.get_question_pooled_embedding(question_id,question_segments,question_attn_mask)
        context_pooled_out = self.get_context_pooled_embedding(context_ids,context_segments,context_attn_mask)

        return question_pooled_out,context_pooled_out

class BiEncoderNllLoss(object):
    def calc(self,
             q_vectors,
             ctx_vectors,
             positive_idx_per_question,
             loss_scale=None):
        scorces = paddle.matmul(q_vectors,paddle.transpose(ctx_vectors,[0,1]))#这里需要对照一下paddle和torch的算子的差异

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scorces.view(q_num, -1)

        softmax_scorces = F.log_softmax(scores,axis=1)

        loss = F.nll_loss(softmax_scorces,paddle.to_tensor(positive_idx_per_question).todevice)#to_device这里需要修改

        max_score = paddle.max(softmax_scorces,axis=1)
        correct_predictions_count = ()#需要修改

        if loss_scale:
            loss.mul_(loss_scale)#对照paddle

        return loss,correct_predictions_count