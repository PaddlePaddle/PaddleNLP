import copy
import json
import os
import re
import sys
import argparse

import paddle

from tqdm import tqdm
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer



def main(
        base_model: str = "",
        share_gradio: bool = False,
):
    args = parse_args()

    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        #prompt = generate_prompt(instruction, input)
        conversation_result: list[tuple[list[int], list[int]]] = tokenizer.encode_chat_inputs(
            [[instruction, 'aa']],
        )
        inputs = {}
        inputs['input_ids'] = paddle.to_tensor([conversation_result['conversations'][0][0]], dtype=paddle.int64)
        seq_length = len(inputs['input_ids'][0])
        #inputs["attention_mask"] = paddle.tril(paddle.ones([seq_length, seq_length], dtype=bool))
        inputs["attention_mask"] = paddle.ones([1, seq_length], dtype=paddle.int64)

        tokenizer.eos_token = '<|eot_id|>'
        with paddle.no_grad():
            generation_output = model.generate(**inputs, max_new_tokens=max_new_tokens, decode_strategy="greedy_search", eos_token_id=[tokenizer.eos_token_id, tokenizer.start_header_id, tokenizer.end_header_id], skip_special_tokens=True)
        s = generation_output[0][0]
        output = tokenizer.decode(s[:-1])
        #return output.split("### Response:")[1].strip()
        return output

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """
    save_file = f'../../../experiment/{args.model}.json'
    create_dir('../../../experiment/')

    dataset = load_data(args)
    tokenizer, model = load_model(args)
    total = len(dataset)
    correct = 0
    miss = 0.001
    output_data = []
    pbar = tqdm(total=total)
    for idx, data in enumerate(dataset):
        instruction = data.get('src')

        predict = evaluate(instruction)
        label = data.get('tgt')
        flag = False
        #predict = extract_answer_letter(args, outputs)
        print(predict)
        print(label)
        if len(predict.split()) > 1 and label.split()[-1] == predict.split()[-1]:
            print(predict.split()[-1])
            correct += 1
            flag = True
        new_data = copy.deepcopy(data)
        new_data['output_pred'] = predict
        new_data['pred'] = predict
        new_data['flag'] = flag
        output_data.append(new_data)
        print(' ')
        print('---------------')
        print('prediction:', predict)
        print('label:', label)
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'{args.dataset}'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_datas = []
    with open(file_path, 'r') as f:
        for line in f:
            json_data = json.loads(line)
            json_datas.append(json_data)
    return json_datas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--model', required=True)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
    ) # fix zwq

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


if __name__ == "__main__":
    main()
