import argparse
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    ChatGLMv2Tokenizer,
    LlamaTokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)
from utils import init_chat_template

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--device', type=str, default="gpu")
    parser.add_argument('--dtype', type=str, default="flaot16")
    parser.add_argument('--approx_model_name', type=str, default="meta-llama/Llama-2-7b-chat")
    parser.add_argument('--target_model_name', type=str, default="meta-llama/Llama-2-7b-chat")
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    return args

def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


def preprocess_text(approx_model_name, source_text):
    tokenizer = AutoTokenizer.from_pretrained(
        approx_model_name,
    )
    # init chat_template for tokenizer
    init_chat_template(tokenizer, approx_model_name)

    if isinstance(tokenizer, LlamaTokenizer) and not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token
    
    # config = AutoConfig.from_pretrained(approx_model_name)

    tokenized_source = tokenizer(
        source_text,
        max_length=None,
        truncation=True,
        truncation_side="left",
        return_tensors="pd",
        padding=True,
        # when use chat_template, it should not add special tokens
        # chatglm2 prefix-tokens can not be tokenized into ids
        add_special_tokens=True,
    )

    decoded_predictions = tokenizer.batch_decode(
            tokenized_source["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    return tokenized_source, decoded_predictions

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma=4):
    tokenized_source, decoded_predictions = preprocess_text(approx_model_name, input_text) # dict[str, paddle.Tensor]
    print("input_text:", input_text)
    print("tokenized_source:", tokenized_source)
    print("decoded_predictions:", decoded_predictions)





if __name__ == '__main__':
    args = parse_arguments()
    generate(args.input, args.approx_model_name, args.target_model_name, args.max_tokens, args.gamma)