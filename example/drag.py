
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import paddle
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = "gpu:1"

paddle.set_device(device)

# import jsonlines
# import paddlenlp.transformers
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

import random
# import torch
import numpy as np
from tqdm import tqdm
import json
import argparse
import re
from tqdm import tqdm
import time
from paddlenlp.drag.utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from paddlenlp.drag.metrics import match, accuracy

#
seed = 633
# os.environ['TORCH_NCCL_AVOID_RECORD_STREAMS'] = '0'

# torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)




model_path = '/root/zzg/self-rag-main/model/llama3-8b-instruct-paddle'
model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model_path = '/root/zzg/llama3-8b-instruct'
# data_path = '/data/home/scv6140/run/hs/self-rag/retrieval_lm/eval_data/popqa/popqa_part_1.jsonl'
# data_path = '/data/home/scv6140/run/hs/self-rag/retrieval_lm/eval_data/popqa/popqa_part_2.jsonl'
# data_path = '/data/home/scv6140/run/hs/self-rag/retrieval_lm/eval_data/popqa/popqa_part_3.jsonl'
# data_path = '/data/home/scv6140/run/hs/self-rag/retrieval_lm/eval_data/popqa/popqa_part_4.jsonl'
# data_path = '/root/zzg/self-rag-main/datasets/eval_data/popqa_longtail.jsonl'
# data_path = '/data/home/scv6140/run/hs/self-rag/retrieval_lm/eval_data/popqa_longtail_retrieval_20.jsonl'
# data_path = '/root/zzg/self-rag-main/datasets/eval_data/popqa_longtail.jsonl'
# data_path = '/data/home/scv6140/run/hs/self-rag/retrieval_lm/eval_data/triviaqa_retrieval.jsonl'
# data_path = '/data/home/scv6140/run/hs/self-rag/retrieval_lm/eval_data/triviaqa_test.jsonl'
# data_path = '/data/home/scv6140/run/hs/self-rag/retrieval_lm/eval_data/test.jsonl'
out_path = '/root/zzg/self-rag-main/retrieval_lm/output/tmp.json'
metric = 'match'
ndocs = 10
show_prev = False
# Qwen = "/root/zzg/self-rag-main/model/Qwen2-0.5B-instruct-paddle"
Qwen = "/root/zzg/self-rag-main/model/Qwen2-0.5B-instruct-paddle-v1"
# Qwen = "/root/zzg/self-rag-main/model/llama3-8b-instruct-paddle"
llama3_Tokenizer = AutoTokenizer.from_pretrained(model_path)
# ---------------------------------------------------------------------------------------don't modify content before this line
wo_decoding = False
wo_anayzer = False
w_irr_fix1 = False
use_conf = False

qwen_model = AutoModelForCausalLM.from_pretrained(Qwen)
qwen_tokenizer = AutoTokenizer.from_pretrained(Qwen)


def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer

def extract_elements(question, max_new_tokens=125):
    sys_instruction = "You are an assistant in extracting key elements from a given question."
    user_instruction = "Question: "
    messages = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": user_instruction + '\n' + question}
    ]

    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pd")

    # Generate response
    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    # print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))

    # Process the generated output
    response = qwen_tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
    print("ELEMENTS: " + response, flush=True)
    return response

alpha_beta = None
def process_explanation(text):
    essential_scores = list(map(float, re.findall(r'\(essential\): (\d+\.\d+)', text, re.IGNORECASE)))
    initial_scores = list(map(float, re.findall(r'\(initial\): (\d+\.\d+)', text, re.IGNORECASE)))
    supplementary_scores = list(map(float, re.findall(r'\(supplementary\): (\d+\.\d+)', text, re.IGNORECASE)))

    if alpha_beta is not None:
        alpha, beta = alpha_beta
        score_max = len(essential_scores) + alpha * len(initial_scores) + beta * len(supplementary_scores)
        std_max = len(essential_scores) + 0.6 * len(initial_scores) + 0.3 * len(supplementary_scores)
        score = sum(essential_scores) + alpha * sum(initial_scores) + beta * sum(supplementary_scores)
        # print(len(essential_scores), len(initial_scores), len(supplementary_scores), score_max)
        try:
            final_score = score * std_max / score_max
        except:
            final_score = score
    else:
        final_score = sum(essential_scores) + 0.6 * sum(initial_scores) + 0.3 * sum(supplementary_scores)
    lex_diver = len(essential_scores) + 1.2 * len(initial_scores) + len(supplementary_scores)
    return final_score, lex_diver

def score_paragraph(question, paragraph, elements, max_new_tokens=125):
    sys_instruction = "You are an assistant in scoring paragraphs based on a given question and its associated elements."
    user_instruction = "Question:\n Elements;\n Paragraphs:"

    paragraph_text = f"{paragraph['title']}\n{paragraph['text']}"

    my_input = (
        f"### Question: {question}\n"
        f"### Element: {elements}\n"
        f"### Paragraphs: {paragraph_text}\n"
    )

    messages = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": user_instruction + '\n' + my_input}
    ]

    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(text)
    model_inputs = qwen_tokenizer([text], return_tensors="pd")

    # Generate response
    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )

    # Process the generated output
    response = qwen_tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
    print("SCORES: " + response, flush=True)
    score, lex_diver = process_explanation(response)
    return (score, lex_diver)

def sort_para(question, paragraphs, max_new_tokens=125):
    # paragraphs = paragraphs[:1]
    elements = extract_elements(question, max_new_tokens)
    scored_texts = [
        (para, *score_paragraph(question, para, elements, max_new_tokens))
        for para in paragraphs
    ]
    lex_diver = scored_texts[0][2] if len(scored_texts) != 0 else None
    scored_texts = [(para, score) for para, score, ld in scored_texts if score is not False]
    sorted_texts = sorted(scored_texts, key=lambda x: x[1], reverse=True)
    # return [text for text, score in sorted_texts[:5]], [text for text, score in sorted_texts[-1:]]
    return sorted_texts[:5], sorted_texts[-1:], lex_diver


def get_score_unsort(question, paragraphs, max_new_tokens=125):
    elements = extract_elements(question, max_new_tokens)
    scored_texts = [
        (para, *score_paragraph(question, para, elements, max_new_tokens))
        for para in paragraphs[:5]
    ]
    lex_diver = scored_texts[0][2] if len(scored_texts) != 0 else None
    scored_texts = scored_texts[:5]
    scores = []
    for para, score, ld in scored_texts:
        scores.append(score if score is not False else 0.)
    return scores, lex_diver

def my_model_rank(my_input, evidences, model, tokenizer, max_new_tokens=50):
    # subquery_num, subquery = Generate_Subquery(my_input, model, max_new_tokens)
    if 1:
        # extracted_info = generate_Triplet(my_input, max_new_tokens)
        # subject = extracted_info['subject']
        # relationship = extracted_info['relationship']

        paddle.device.cuda.synchronize()
        part1_time = time.perf_counter()
        relevant_para = []
        irrelevant_para = []
        results = {}
        decoding_flag = not wo_decoding


        if not wo_anayzer:
            # evidences, irr_evidences=sort_para(my_input,evidences)
            top5, bottom1, lex_diver = sort_para(my_input,evidences)
            evidences = [text for text, score in top5]
            scores = [score for text, score in top5]
            # if top5[0][1] < 1.6 and bottom1[0][1] < 1:
            if 1:
                irr_evidences = [text for text, score in bottom1]
        else:
            if decoding_flag:
                # TODO
                scores, lex_diver = get_score_unsort(my_input, evidences)
                pass
            evidences = evidences[:5]
            irr_evidences = evidences[-1:]
        if decoding_flag:
            model.config.decode_strategy = "sampling_irr"
            model.generation_config.decode_strategy = "sampling_irr"
            set_para_ano(
                lex_diver if not use_conf else -1,
                scores
            )


        if w_irr_fix1:
            # irr_evidences=[
            #     {"title": "Ethnic groups in Rwanda", "text": "divert the emphasis from ethnicity to a division of the population into categories of victim, victors, survivors, and perpetrators. However, in identifying victims and survivors, some Rwandans are left to be identified as perpetrators. This becomes increasingly problematic as all Hutus are deemed perpetrators—where their survival of the genocide seems to imply some form of complicity with the former government. Thus, in this process of rebuilding and bringing guilty parties to justice, the current government is providing dangling linkages back to the very ethnicities they wish to abolish and is risking further entrenching supposed “past” ethnic divisions. Furthermore, government policy"}
            # ]
            irr_evidences =[
              {"title": "Rebirth (Buddhism)", "text": "Rebirth (Buddhism) Rebirth in Buddhism refers to its teaching that the actions of a person lead to a new existence after death, in endless cycles called \"saṃsāra\". This cycle is considered to be \"dukkha\", unsatisfactory and painful. The cycle stops only if liberation is achieved by insight and the extinguishing of desire. Rebirth is one of the foundational doctrines of Buddhism, along with Karma, nirvana and moksha. The rebirth doctrine in Buddhism, sometimes referred to as reincarnation or metempsychosis, asserts that rebirth does not necessarily take place as another human being, but as an existence in one of the six"}
            ]
        for evidence in evidences:
            relevant_para.append("[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(evidence["title"], evidence["text"]))


        paddle.device.cuda.synchronize
        part1_time = time.perf_counter() - part1_time
        # relevant_para = relevant_para[:5]

        paddle.device.cuda.synchronize
        part2_time = time.perf_counter()
        if not relevant_para:
            sys_msg=PROMPT_DICT["prompt_for_combine_no_retrieval"][0]['content']
            user_msg=PROMPT_DICT["prompt_for_combine_no_retrieval"][1]['content'].format(instruction=my_input)
            msgs = [
                {"role":"system","content":sys_msg},
                {"role":"user","content":user_msg}
            ]
            prompt = llama3_Tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        else:
            sys_msg=PROMPT_DICT["prompt_for_combine"][0]['content']
            user_msg=PROMPT_DICT["prompt_for_combine"][1]['content'].format(instruction=my_input,paragraphs='\n'.join(relevant_para))
            msgs = [
                {"role":"system","content":sys_msg},
                {"role":"user","content":user_msg}
            ]
            prompt = llama3_Tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        token_inputs = tokenizer([prompt], return_tensors="pd")

        inputs_ids_irr = None


        if decoding_flag and lex_diver is not None:
            for irr_evidence in irr_evidences:
                irrelevant_para.append("[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(irr_evidence["title"], irr_evidence["text"]))
            sys_msg_irr = PROMPT_DICT["prompt_for_combine"][0]['content']
            user_msg_irr = PROMPT_DICT["prompt_for_combine"][1]['content'].format(instruction=my_input,paragraphs='\n'.join(irrelevant_para))
            # user_msg_irr = PROMPT_DICT["prompt_for_combine"][1]['content'].format(instruction=my_input,subject=subject,relationship=relationship,paragraphs=irrelevant_para[0])
            msgs_irr = [
                {"role":"system","content":sys_msg_irr},
                {"role":"user","content":user_msg_irr}
            ]
            prompt_irr = llama3_Tokenizer.apply_chat_template(msgs_irr, add_generation_prompt=True, tokenize=False)
            # prompt_irr = PROMPT_DICT['prompt_no_input_retrieval'].format(paragraph='\n'.join(irrelevant_para),instruction=my_input)
            token_inputs_irr = tokenizer([prompt_irr], return_tensors="pd")
            inputs_ids_irr = token_inputs_irr.input_ids

            # 计算question（含）之前的token长度
            question_only_msg = PROMPT_DICT["prompt_for_combine"][1]['content'].format(instruction=my_input,paragraphs='\n')
            question_msgs = [
                {"role":"system","content":sys_msg},
                {"role":"user","content":question_only_msg}
            ]
            question_only_prompt = llama3_Tokenizer.apply_chat_template(question_msgs, add_generation_prompt=True, tokenize=False)
            question_token_inputs = tokenizer([question_only_prompt], return_tensors="pd")
            question_token_len = question_token_inputs.input_ids.shape[1]


            question_and_doc_len = []
            for i in range(len(relevant_para)):
                _sys_msg = PROMPT_DICT["prompt_for_combine"][0]['content']
                _user_msg = PROMPT_DICT["prompt_for_combine"][1]['content'].format(instruction=my_input,paragraphs='\n'.join(relevant_para[:(i + 1)]))
                _msgs = [
                    {"role":"system","content":_sys_msg},
                    {"role":"user","content":_user_msg}
                ]
                _prompt = llama3_Tokenizer.apply_chat_template(_msgs, add_generation_prompt=True, tokenize=False)

                _token_inputs = tokenizer([_prompt], return_tensors="pd")

                question_and_doc_len.append(_token_inputs.input_ids.shape[1])

            print(question_token_len, question_and_doc_len, flush=True)
            generated_ids = model.generate(
                **token_inputs,
                max_new_tokens=512,
                use_irr = decoding_flag,
                inputs_irr=inputs_ids_irr,
                alpha=3,
                question_token_len = question_token_len,
                question_and_doc_len = question_and_doc_len,
            )
            # generated_ids = [
                # output_ids[len(input_ids):] for input_ids, output_ids in zip(token_inputs.input_ids, generated_ids)
            # ]
            output_text = tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
        else:
            generated_ids_prev = model.generate(
                **token_inputs,
                max_new_tokens=512,
            )
            # generated_ids_prev = [
                # output_ids[len(input_ids):] for input_ids, output_ids in zip(token_inputs.input_ids, generated_ids_prev)
            # ]
            output_text_prev = tokenizer.batch_decode(generated_ids_prev[0], skip_special_tokens=True)[0]
            output_text = output_text_prev

        paddle.device.cuda.synchronize
        part2_time = time.perf_counter() - part2_time

        print('-------------------------------PROMPT-------------------------------', flush=True)
        print(prompt, flush=True)
        if 'prompt_irr' in locals().keys():
            print('-----------------------------IRR_PROMPT-----------------------------', flush=True)
            print(prompt_irr, flush=True)

        print(">>>>>>>>>>>>>>>OUTPUT:", output_text, flush=True)
        print(f"this_time: {part1_time} | {part2_time}", flush=True)
        # print("OUTPUT_TOKEN:", generated_ids or generated_ids_prev, flush=True)

    return output_text, results, part1_time, part2_time

def process_data_evidences(demonstration, top_n):
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    # prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return ctx_key, evidences


def preprocess_input_data(dataset, task=None):
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        else:
            prompt = instruction + "\n\n## Input:\n\n" + \
                item["question"] if instruction is not None else item["question"]
            item["instruction"] = prompt
        new_data.append(item)

    return new_data


from paddlenlp.drag.drag_generate import set_para, set_para_ano, Drag
# def set_para(*args):
    # pass
# def set_para_ano(*args):
    # pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str, default='/data/home/scv6140/run/hs/self-rag/retrieval_lm/eval_data/triviaqa/triviaqa_part_1.jsonl')
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument("--ndocs", type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    # Decoding hyperparams
    parser.add_argument('--threshold', type=float,
                        default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true",
                        help="use ground score")
    parser.add_argument(
        "--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width",  type=int,
                        default=2, help="beam search width")
    parser.add_argument("--max_depth",  type=int,
                        default=2, help="tree depth width")
    parser.add_argument("--w_rel",  type=float, default=1.0,
                        help="reward weight for document relevance")
    parser.add_argument("--w_sup",  type=float, default=1.0,
                        help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use",  type=float, default=1.0,
                        help="reward weight for overall completeness / utility.")
    parser.add_argument('--mode', type=str, help="mode to control retrieval.",
                        default="default", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieve'],)
    parser.add_argument('--metric', type=str, help="metric to be used during evaluation")
    parser.add_argument("--thresh",  type=float, default=0.8)
    parser.add_argument("--use_conf", action="store_true")
    parser.add_argument('--extra', type=str, default='')
    parser.add_argument('--alpha_beta', type=str, default=None)
    parser.add_argument("--wo_dec", action="store_true")
    parser.add_argument("--wo_ana", action="store_true")
    parser.add_argument("--start_round", type=int, default=0)
    args = parser.parse_args()

    ## debug
    args.model_name = model_path
    # args.input_file = data_path
    if args.alpha_beta is not None:
        alpha, beta = args.alpha_beta.split('_', 1)
        args.extra += f"alpha{alpha}_beta{beta}"
        try:
            global alpha_beta
            alpha_beta = float(alpha), float(beta)
        except:
            print("alpha_beta format wrong, which is", args.alpha_beta)
            exit()

    if args.wo_dec:
        args.extra += f"_nodec"
        global wo_decoding
        wo_decoding = True
    if args.wo_ana:
        args.extra += f"_noana"
        global wo_anayzer
        wo_anayzer = True
    if args.start_round != 0:
        args.extra += f"_st_round_{args.start_round}"


    args.max_new_tokens = 100
    args.output_file = out_path
    args.metric = 'match'
    args.ndocs = ndocs
    args.dtype = 'half'
    print(args, flush=True)
    set_para(args.thresh, "full", 0, 0, 0, False, args.extra)
    if args.use_conf:
        global use_conf
        use_conf = True

    gpt = args.model_name
    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    input_data = preprocess_input_data(
        input_data, task=args.task)

    tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(gpt, low_cpu_mem_usage=True)
    print(model.__class__)
    model.__class__ = Drag
    model.check()


    def generate(prompt, evidences, max_new_tokens):
        return my_model_rank(prompt, evidences, model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)

    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0
    sum_time = [0, 0]
    run_round = 0
    missing_round = []
    for i, row in tqdm(enumerate(input_data)):
        if i < args.start_round:
            continue
        results = {}
        my_input = row['instruction']
        _, evidences = process_data_evidences(row, top_n=args.ndocs)

        # try:
            # pred, results, t1, t2 = generate(
                # my_input, evidences, max_new_tokens=args.max_new_tokens,)
        # except:
            # missing_round.append(i)
            # print("MISSING ROUND:", missing_round, flush=True)
            # continue


        pred, results, t1, t2 = generate(
            my_input, evidences, max_new_tokens=args.max_new_tokens,)

        sum_time[0] += t1
        sum_time[1] += t2
        run_round += 1
        print(f"avg_time: {sum_time[0] / run_round} | {sum_time[1] / run_round}", flush=True)
        if type(pred) is str and len(pred)>0 and (pred[0] == "#" or pred[0] == ":"):
            pred = pred[1:]
        prompts.append(my_input)
        preds.append(pred)
        all_results.append(results)
        # if do_retrieve is True:
        #     count += 1
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        if args.metric == "accuracy":
            metric_result = accuracy(pred, row["output"])

        elif args.metric == "match":
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)
        if i % 10 == 0:
            print("average: {}".format(np.mean(metric_results)), flush=True)
            final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                             "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}
            with open(args.output_file + "_tmp", "w") as outfile:
                json.dump(final_results, outfile)

    final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                     "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}
    with open(args.output_file, "w") as outfile:
        json.dump(final_results, outfile)

    print("Final result: {0}".format(np.mean(metric_results)), flush=True)
    print("MISSING ROUND:", missing_round, flush=True)
    #print("Retrieval Frequencies: {0}".format(count / len(final_results)))
    print(metric_results, flush=True)

if __name__ == "__main__":
    main()
