from paddlenlp.transformers import *
ernie = AutoModel.from_pretrained("ppminilm-6l-768h")
t = AutoTokenizer.from_pretrained("ppminilm-6l-768h")
