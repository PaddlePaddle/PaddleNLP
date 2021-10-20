## bert-base-japanese

基于bert的日语训练模型的相关权重参数，其中包括：

| Pretrained Weight                          | Language | Details of the model                                         |
| ------------------------------------------ | -------- | ------------------------------------------------------------ |
| bert-base-japanese                         | Japanese | 12 repeating layers, 768-hidden, 12-heads. This version of the model processes input texts with word-level  tokenization based on the IPA dictionary, followed by the WordPiece  subword tokenization. [reference](https://huggingface.co/cl-tohoku/bert-base-japanese) |
| bert-base-japanese-char                    | Japanese | 12 repeating layers, 768-hidden, 12-heads. This version of the model processes input texts with word-level  tokenization based on the IPA dictionary, followed by character-level  tokenization. [reference](https://huggingface.co/cl-tohoku/bert-base-japanese-char) |
| bert-base-japanese-char-whole-word-masking | Japanese | 12 repeating layers, 768-hidden, 12-heads. This version of the model processes input texts with word-level  tokenization based on the IPA dictionary, followed by character-level  tokenization. Additionally, the model is trained with the whole word masking enabled  for the masked language modeling (MLM) objective..[reference](https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking) |
| bert-base-japanese-whole-word-masking      | Japanese | 12 repeating layers, 768-hidden, 12-heads. This version of the model processes input texts with word-level  tokenization based on the IPA dictionary, followed by the WordPiece  subword tokenization. Additionally, the model is trained with the whole word masking enabled  for the masked language modeling (MLM) objective. [reference](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking) |

