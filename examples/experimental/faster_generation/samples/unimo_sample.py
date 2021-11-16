from paddlenlp.transformers import UNIMOLMHeadModel, UNIMOTokenizer

model_name = 'unimo-text-1.0-lcsts-new'

model = UNIMOLMHeadModel.from_pretrained(model_name)
tokenizer = UNIMOTokenizer.from_pretrained(model_name)


def post_process_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return tokens


inputs = "深度学习是人工智能的核心技术领域。百度飞桨作为中国首个自主研发、功能丰富、开源开放的产业级深度学习平台,将从多层次技术产品、产业AI人才培养和强大的生态资源支持三方面全面护航企业实现快速AI转型升级。"
print("Model input:", inputs)
inputs = tokenizer.gen_encode(
    inputs,
    add_start_token_for_decoding=True,
    return_tensors=True,
    is_split_into_words=False)
model.eval()
outputs, _ = model.generate(
    input_ids=inputs['input_ids'],
    token_type_ids=inputs['token_type_ids'],
    position_ids=inputs['position_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=64,
    decode_strategy='beam_search',
    num_beams=2)

result = post_process_response(outputs[0].numpy(), tokenizer)
result = "".join(result)
print(result)
# 百度飞桨：深度学习助力企业转型升级
