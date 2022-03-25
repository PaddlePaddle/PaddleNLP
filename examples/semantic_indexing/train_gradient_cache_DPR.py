import paddle
from paddle import nn
from paddle.nn import functional as F
from biencoder_base_model import BiEncoder,BiEncoderNllLoss
from NQdataset import BiEncoderPassage,BiEncoderSample,BiENcoderBatch,BertTensorizer,NQdataSetForDPR,DataUtil
from paddlenlp.transformers.bert.modeling import BertModel

global batch_size
global learning_rate
global weight_decay
global drop_out
global embedding_output_size
global data_path

question_encoder = BertModel.from_pretrained("base")
context_encoder = BertModel.from_pretrained("base")


model = BiEncoder(question_encoder=question_encoder,context_encoder=context_encoder,dropout=drop_out,output_emb_size=embedding_output_size)

dataset = NQdataSetForDPR(data_path)
data_loader = paddle.io.DataLoader(dataset,batch_size=batch_size,shuffle=True)#记得查看collate_fn

optimizer = paddle.optimizer.AdamW(
    learning_rate=learning_rate,
    parameters=model.parameters(),
    weight_decay=weight_decay
)





def train():
    for batch in data_loader:
        output = model(

        )

