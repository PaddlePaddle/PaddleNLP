import argparse
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoTokenizer
import paddle
import time

def parse_args(prog=None):
    """
    parse_args
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--model_name_or_path", type=str, help="model name or local path", required=True)
    parser.add_argument('--do_forward', action='store_true', help='fowrward test')
    parser.add_argument('--do_generate', action='store_true', help='generate test')
    return parser.parse_args()

@paddle.no_grad()
def predict_generate(model, inputs):
    for i in range(10):
        start = time.perf_counter()
        result = model.generate(**inputs, 
                    max_length=100,
                    decode_strategy="greedy_search",
                    use_cache=True,
                    )
        hf_cost = (time.perf_counter() - start) * 1000
        print("Speed test:", hf_cost)
        infer_data = result[0]
        for x in infer_data.tolist():
            res = tokenizer.decode(x, skip_special_tokens=True)
            print(res)

@paddle.no_grad()
def predict_forward(model, inputs):
    for i in range(10):
        start = time.perf_counter()
        result = model(**inputs)
        hf_cost = (time.perf_counter() - start) * 1000
        print("Speed test:", hf_cost)


if __name__ == "__main__":
    args = parse_args()
    all_texts = [
        "你好",
        "去年9月，拼多多海外版“Temu”正式在美国上线。数据显示，截至2023年2月23日，Temu App新增下载量4000多万，新增用户增速第一，AppStore购物榜霸榜69天、Google Play购物榜霸榜114天。",
    ]
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_state_as_np=True,
                low_cpu_mem_usage=True,
            )
    model.eval()

    if args.do_forward:
        for input_text in all_texts:
            print(f"text: {input_text}")
            inputs = tokenizer([input_text], return_tensors="pd",return_token_type_ids=False)
            predict_forward(model, inputs)

    if args.do_generate:
        for input_text in all_texts:
            print(f"text: {input_text}")
            _inputs = tokenizer(
                input_text,
                padding=True,
                return_tensors="np",
                max_length=50,
                return_attention_mask=True,
                return_position_ids=True,
            )
            inputs_tensor = {}
            for key, value in _inputs.items():
                inputs_tensor[key] = paddle.to_tensor(value)
            predict_generate(model, inputs_tensor)
            