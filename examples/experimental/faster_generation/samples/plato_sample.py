from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer

model_name = 'plato-mini'

tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name)
model = UnifiedTransformerLMHeadModel.from_pretrained(model_name)


def post_process_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return tokens


inputs = ['你好啊，你今年多大了']
print("Model input:", inputs)
inputs = tokenizer.dialogue_encode(
    inputs,
    add_start_token_as_response=True,
    return_tensors=True,
    is_split_into_words=False)

outputs, _ = model.generate(
    input_ids=inputs['input_ids'],
    token_type_ids=inputs['token_type_ids'],
    position_ids=inputs['position_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=64,
    decode_strategy='sampling',
    top_k=5)

result = post_process_response(outputs[0].numpy(), tokenizer)
result = "".join(result)
print(result)
# 我今年23岁了,你今年多大了?
