## uer/roberta-base-finetuned-chinanews-chinese
在chinanews数据集上微调的新闻分类模型

```python
tokenizer = RobertaTokenizer.from_pretrained('uer/roberta-base-finetuned-chinanews-chinese')
token = tokenizer('北京上个月召开了两会')
# print(token)
config = RobertaModel.pretrained_init_configuration[
    'uer/roberta-base-finetuned-chinanews-chinese']
roberta = RobertaModel(**config)
model = RobertaForSequenceClassification(roberta, 7)
model_state = paddle.load(os.path.join(path, "model_state.pdparams"))
model.load_dict(model_state)
model.eval()
input_ids = paddle.to_tensor(token['input_ids'], dtype='int64').unsqueeze(0)
with paddle.no_grad():
    output = model(input_ids)
import paddle.nn.functional as F
output = F.softmax(output)
id2label = {
    "0": "mainland China politics",
    "1": "Hong Kong - Macau politics",
    "2": "International news",
    "3": "financial news",
    "4": "culture",
    "5": "entertainment",
    "6": "sports"
}
for i in range(7):
    print("{}: \t {}".format(id2label[str(i)], output[0][i].item()))

'''
mainland China politics:      0.7211663722991943
Hong Kong - Macau politics:      0.0015174372820183635
International news:      0.00030049943597987294
financial news:      0.1901436597108841
culture:      0.0707709938287735
entertainment:      0.007615766953676939
sports:      0.008485228754580021
'''
