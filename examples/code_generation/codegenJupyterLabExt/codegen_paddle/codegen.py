import time
from paddlenlp.utils.log import logger


def gen_code(prompt: str, model, tokenizer, generate_config) -> str:
    if model is None or tokenizer is None or generate_config is None:
        return ''
    start_time = time.time()
    logger.info("Start generating code")
    tokenized = tokenizer(prompt,
                          truncation=True,
                          return_tensors='pd')
    output, _ = model.generate(
        tokenized["input_ids"],
        max_length=16,
        min_length=generate_config.min_length,
        decode_strategy=generate_config.decode_strategy,
        top_k=generate_config.top_k,
        repetition_penalty=generate_config.repetition_penalty,
        temperature=generate_config.temperature,
        use_faster=generate_config.use_faster,
        use_fp16_decoding=generate_config.use_fp16_decoding)
    logger.info("Finish generating code")
    end_time = time.time()
    logger.info(f"Time cost: {end_time - start_time}")
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Generated code: {output}")
    return output
