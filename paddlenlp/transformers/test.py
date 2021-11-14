from paddlenlp.transformers import *

bert = BertModel.from_pretrained(
    'iverxin/bert-base-japanese-char-whole-word-masking')
print(bert)

orderdict = bert.state_dict()
i = 0
total_sum = 0
for key in orderdict.keys():
    i += 1
    #print(key)
    tensor_sum = 1
    for j in range(len(orderdict[key].shape)):
        tensor_sum = orderdict[key].shape[j] * tensor_sum
    #print(tensor_sum)
    total_sum += tensor_sum
print(total_sum / 1000000)
