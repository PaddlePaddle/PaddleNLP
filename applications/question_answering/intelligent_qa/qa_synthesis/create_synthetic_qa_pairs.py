import os
import json
import sys
import argparse
import re
from tqdm import tqdm

from paddlenlp import Taskflow


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--answer_generation_model_path', 
                        type=str, 
                        default=None,
                        help='the model path to be loaded for question_generation taskflow')
    parser.add_argument('--question_generation_model_path', 
                        type=str, 
                        default=None,
                        help='the model path to be loaded for question_generation taskflow')
    parser.add_argument('--filtration_model_path', 
                        type=str, 
                        default=None,
                        help='the model path to be loaded for filtration taskflow')
    parser.add_argument('--source_file_path',
                        type=str,
                        default=None,
                        help='the souce file path')
    parser.add_argument('--target_file_path',
                        type=str,
                        default=None,
                        help='the target json file path')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='the batch size when using taskflow')
    parser.add_argument("--do_debug", 
                        action='store_true', 
                        help="Whether to do debug")

    parser.add_argument('--prompt',
                        type=str,
                        default=None,
                        help='the prompt when using taskflow, seperate by ,')
    parser.add_argument('--position_prob',
                        type=float,
                        default=0.01,
                        help='the batch size when using taskflow')
    parser.add_argument('--max_answer_candidates', 
                        type=int, 
                        default=5,
                        help='the max number of return answer candidate for each input')

    parser.add_argument('--num_return_sequences', 
                        type=int, 
                        default=3,
                        help='the number of return sequences for each input sample, it should be less than num_beams') 
    parser.add_argument('--max_question_length', 
                        type=int, 
                        default=50,
                        help='the max decoding length')
    parser.add_argument('--decode_strategy',
                        type=str,
                        default=None,
                        help='the decode strategy')
    parser.add_argument('--num_beams', 
                        type=int, 
                        default=6,
                        help='the number of beams when using beam search')
    parser.add_argument('--num_beam_groups', 
                        type=int, 
                        default=1,
                        help='the number of beam groups when using diverse beam search')
    parser.add_argument('--diversity_rate', 
                        type=float, 
                        default=0.0,
                        help='the diversity_rate when using diverse beam search')
    parser.add_argument('--top_k', 
                        type=float, 
                        default=0,
                        help='the top_k when using sampling decoding strategy')
    parser.add_argument('--top_p', 
                        type=float, 
                        default=1.0,
                        help='the top_p when using sampling decoding strategy')
    parser.add_argument('--temperature', 
                        type=float, 
                        default=1.0,
                        help='the temperature when using sampling decoding strategy')

    parser.add_argument("--do_filtration", 
                        action='store_true', 
                        help="Whether to do filtration")
    parser.add_argument('--filtration_position_prob',
                        type=float,
                        default=0.1,
                        help='the batch size when using taskflow')
    args = parser.parse_args()
    return args



def answer_generation_from_paragraphs(paragraphs, batch_size=16, model=None, max_answer_candidates=5, schema=None, wf=None):
    """Generate answer from given paragraphs."""
    result = []
    buffer = []
    i = 0
    len_paragraphs = len(paragraphs)
    for paragraph_tobe in tqdm(paragraphs):
        buffer.append(paragraph_tobe)
        if len(buffer) == batch_size or (i+1) == len_paragraphs:
            predicts = model(buffer)
            paragraph_list = buffer
            buffer = []
            for predict_dict, paragraph in zip(predicts, paragraph_list):
                answers = []
                probabilitys = []
                for prompt in schema:
                    if prompt in predict_dict:
                        answer_dicts = predict_dict[prompt]
                        answers += [answer_dict['text'] for answer_dict in answer_dicts]
                        probabilitys += [answer_dict['probability'] for answer_dict in answer_dicts]
                    else:
                        answers += []
                        probabilitys += []
                candidates = sorted([(a, p) for a, p in zip(answers, probabilitys)], key=lambda x:-x[1])
                if len(candidates) > max_answer_candidates:
                    candidates = candidates[:max_answer_candidates]
                outdict = {
                    'context': paragraph,
                    'answer_candidates': candidates,
                }
                if wf:
                    wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")
                result.append(outdict)
        i += 1
    return result


def create_fake_question(json_file_or_pair_list, out_json=None, num_return_sequences=1, all_sample_num=None,  batch_size=8):
    if out_json:
        wf = open(out_json, 'w', encoding='utf-8')
    if isinstance(json_file_or_pair_list, list):
        all_lines = json_file_or_pair_list
    else:
        rf = open(json_file_or_pair_list, 'r', encoding='utf-8')
        all_lines = []
        for json_line in rf:
            line_dict = json.loads(json_line)
            all_lines.append(line_dict)
        rf.close()
    num_all_lines = len(all_lines)
    output = []
    context_buffer = []
    answer_buffer = []
    answer_probability_buffer = []
    true_question_buffer = []
    i =  0
    for index, line_dict in enumerate(tqdm(all_lines)):
        if 'question' in line_dict:
            q = line_dict['question']
        else:
            q = ''
        c = line_dict['context']
        assert 'answer_candidates' in line_dict
        answers = line_dict['answer_candidates']
        if not answers:
            continue
        for j, pair in enumerate(answers):
            a,p = pair 
            context_buffer += [c] 
            answer_buffer += [a]
            answer_probability_buffer += [p]
            true_question_buffer += [q]
            if (i+1) % batch_size == 0 or (all_sample_num and (i+1) == all_sample_num) or ((index+1) == num_all_lines and j == len(answers)-1):  
                result_buffer = question_generation([{'context':context, 'answer':answer} for context, answer in zip(context_buffer, answer_buffer)])
                context_buffer_temp, answer_buffer_temp, answer_probability_buffer_temp, true_question_buffer_temp = [], [], [], []
                for context, answer, answer_probability, true_question in zip(context_buffer, answer_buffer, answer_probability_buffer, true_question_buffer):
                    context_buffer_temp += [context] * num_return_sequences
                    answer_buffer_temp += [answer] * num_return_sequences
                    answer_probability_buffer_temp += [answer_probability] * num_return_sequences
                    true_question_buffer_temp += [true_question] * num_return_sequences
                result_one_two_buffer = [(one, two) for one,two in zip(result_buffer[0], result_buffer[1])]
                for context, answer, answer_probability, true_question, result in zip(context_buffer_temp, answer_buffer_temp, answer_probability_buffer_temp,  true_question_buffer_temp, result_one_two_buffer):
                    fake_quesitons_tokens = [result[0]]
                    fake_quesitons_scores = [result[1]]
                    for fake_quesitons_token, fake_quesitons_score in zip(fake_quesitons_tokens, fake_quesitons_scores):
                        out_dict = {'context':context, 'synthetic_answer':answer, 'synthetic_answer_probability':answer_probability, 'synthetic_question':fake_quesitons_token, 'synthetic_question_probability':fake_quesitons_score, 'true_question':true_question, }
                        if out_json:
                            wf.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                        output.append(out_dict)
                context_buffer = []
                answer_buffer = []
                true_question_buffer = []
            if all_sample_num and (i+1) >= all_sample_num:
                break
            i += 1
    if out_json:
        wf.close()
    return output

def filtration(paragraphs, batch_size=16, model=None, schema=None, wf=None, wf_debug=None):
    result = []
    buffer = []
    valid_num, invalid_num = 0, 0
    i = 0
    len_paragraphs = len(paragraphs)
    for paragraph_tobe in tqdm(paragraphs):
        buffer.append(paragraph_tobe)
        if len(buffer) == batch_size or (i+1) == len_paragraphs:
            model_inputs = []
            for d in buffer:
                context = d['context']
                synthetic_question = d['synthetic_question']
                prefix = '问题：' + synthetic_question + '上下文：'
                content = prefix + context
                model_inputs.append(content)
            predicts = model(model_inputs)
            paragraph_list = buffer
            buffer = []
            for predict_dict, paragraph in zip(predicts, paragraph_list):
                context = paragraph['context']
                synthetic_question = paragraph['synthetic_question']
                synthetic_question_probability = paragraph['synthetic_question_probability']
                synthetic_answer = paragraph['synthetic_answer']
                synthetic_answer_probability = paragraph['synthetic_answer_probability']

                answers = []
                probabilitys = []
                for prompt in schema:
                    if prompt in predict_dict:
                        answer_dicts = predict_dict[prompt]
                        answers += [answer_dict['text'] for answer_dict in answer_dicts]
                        probabilitys += [answer_dict['probability'] for answer_dict in answer_dicts]
                    else:
                        answers += []
                        probabilitys += []
                candidates = [an for an, pro in sorted([(a,p) for a, p in zip(answers, probabilitys)], key=lambda x:-x[1])]
                out_dict = {'context':context, 'synthetic_answer':synthetic_answer, 'synthetic_answer_probability':synthetic_answer_probability, 'synthetic_question':synthetic_question, 'synthetic_question_probability':synthetic_question_probability, }
                if synthetic_answer in candidates:
                    if wf:
                        wf.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                    result.append(out_dict)
                    valid_num += 1
                else:
                    if wf_debug:
                        wf_debug.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                    invalid_num  += 1
        i += 1
    print('valid synthetic number:', valid_num)
    print('invalid sythetic numbewr:', invalid_num)
    return result

if __name__ == '__main__':
    args = parse_args()
    assert args.prompt
    schema = args.prompt.strip().split(',')
    answer_generator = Taskflow("information_extraction", schema=schema, task_path=args.answer_generation_model_path, batch_size=args.batch_size, position_prob=args.position_prob)
    assert args.source_file_path
    paragraphs = []
    if args.source_file_path.endswith('.json'):
       with open(args.source_file_path, 'r', encoding='utf-8') as rf:
            for json_line in rf:
                line_dict = json.loads(json_line)
                assert 'context' in line_dict or 'content' in line_dict
                if 'context' in line_dict:
                    paragraphs.append(line_dict['context'].strip())
                elif 'content' in line_dict:
                    paragraphs.append(line_dict['content'].strip())
    else:
        with open(args.source_file_path, 'r', encoding='utf-8') as rf:
            for line in rf:
                paragraphs.append(line.strip())

    synthetic_context_answer_pairs = answer_generation_from_paragraphs(paragraphs, batch_size=args.batch_size, model=answer_generator, max_answer_candidates=args.max_answer_candidates, schema=schema, wf=open('/root/project/data/faq-qa/yiqing-answer-agnostic-unsupervised-faq-qa/yiqing_data.json.synthetic_answer.json', 'w', encoding='utf-8'))
    print('create synthetic answers successfully!')
    

    question_generation = Taskflow("question_generation", task_path=args.question_generation_model_path, output_scores=True, max_length=args.max_question_length, is_select_from_num_return_sequences=False, num_return_sequences=args.num_return_sequences,  batch_size=args.batch_size, decode_strategy=args.decode_strategy, num_beams=args.num_beams, num_beam_groups=args.num_beam_groups, diversity_rate=args.diversity_rate, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)
    synthetic_answer_question_pairs = create_fake_question(synthetic_context_answer_pairs, None if args.do_filtration else args.target_file_path, args.num_return_sequences, None, args.batch_size)
    print('create synthetic question-answer pairs successfully!')


    wf = None
    wf_debug = None
    if args.target_file_path:
        wf = open(args.target_file_path, 'w', encoding='utf-8')
        if args.do_debug:
            wf_debug = open(args.target_file_path + '.debug.json', 'w', encoding='utf-8')
    if args.do_filtration:
        filtration_model = Taskflow("information_extraction", schema=['答案'], task_path=args.filtration_model_path, batch_size=args.batch_size, position_prob=args.filtration_position_prob)
        filtration(synthetic_answer_question_pairs, batch_size=16, model=filtration_model, schema=['答案'], wf=wf, wf_debug=wf_debug)
        print('filter synthetic question-answer pairs successfully!')
    rf.close()
    wf.close()