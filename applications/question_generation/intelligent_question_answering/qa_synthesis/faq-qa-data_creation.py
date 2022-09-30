import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--do_create_faq_corpus", 
                        action='store_true', 
                        help="Whether to do create faq corpus, inculding qa_pair.csv, qac_triple.csv, train.csv, q_corpus.csv, dev.csv")
    parser.add_argument('--source_file_path',
                        type=str,
                        default=None,
                        help='the souce json file path')
    parser.add_argument('--target_dir_path',
                        type=str,
                        default=None,
                        help='the target json file path')
    parser.add_argument('--test_sample_num',
                        type=int,
                        default=0,
                        help='the test sample number when convert_json_to_data')
    parser.add_argument('--train_sample_num',
                        type=int,
                        default=0,
                        help='the test sample number when convert_json_to_data')
    parser.add_argument('--all_sample_num',
                        type=int,
                        default=None,
                        help='the all sample number when convert_json_to_data, train_sample_num==all_sample_num-test_sample_num')


    # parser.add_argument("--do_extract_trans_from_fake_question", 
    #                     action='store_true', 
    #                     help="Whether to do extract_trans_from_fake_question")
    # parser.add_argument('--source_trans_json_file_path',
    #                     type=str,
    #                     default=None,
    #                     help='the json source file path for trans file creating')
    # parser.add_argument('--target_trans_txt_file_path',
    #                     type=str,
    #                     default=None,
    #                     help='the txt target file path for trans file creating')
    # parser.add_argument('--query_answer_file_path',
    #                     type=str,
    #                     default=None,
    #                     help='the query-answer file path for extract_trans_from_fake_question')           
    # parser.add_argument("--do_create_test_qq_pair", 
    #                     action='store_true', 
    #                     help="Whether to do create_test_qq_pair")
    # parser.add_argument('--qq_pair_source_ori_file_path',
    #                     type=str,
    #                     default=None,
    #                     help='the original source file path for qq-pair creating')
    # parser.add_argument('--qq_pair_source_trans_file_path',
    #                     type=str,
    #                     default=None,
    #                     help='the translated source file path for qq-pair creating')
    # parser.add_argument('--qq_pair_target_file_path',
    #                     type=str,
    #                     default=None,
    #                     help='the target file path for qq-pair creating')

    args = parser.parse_args()
    return args

def convert_json_to_data(json_file, out_dir, test_sample_num, train_sample_num, all_sample_num=None):
    with open(json_file, 'r', encoding='utf-8') as rf, \
                        open(os.path.join(out_dir, 'qa_pair.csv'), 'w', encoding='utf-8') as qa_pair_wf, \
                        open(os.path.join(out_dir, 'qac_triple.csv'), 'w', encoding='utf-8') as qac_triple_wf, \
                        open(os.path.join(out_dir, 'train.csv'), 'w', encoding='utf-8') as train_wf, \
                        open(os.path.join(out_dir, 'q_corpus.csv'), 'w', encoding='utf-8') as q_corpus_wf, \
                        open(os.path.join(out_dir, 'dev.csv'), 'w', encoding='utf-8') as test_wf:
        for i, json_line in enumerate(rf.readlines()):
            line_dict = json.loads(json_line)
            context = line_dict['context']
            if 'answer' in line_dict and 'question' in line_dict:
                answer = line_dict['answer']
                question = line_dict['question']
            elif 'synthetic_answer' in line_dict and 'synthetic_question' in line_dict:
                answer = line_dict['synthetic_answer']
                question = line_dict['synthetic_question']

            if isinstance(question , list):
                question = question[0]
            else:
                question = question
            
            
            
            if i < test_sample_num:
                test_wf.write(question.replace('\n', ' ').replace('\t', ' ').strip() + '\n')
            elif test_sample_num<= i < test_sample_num + train_sample_num:
                train_wf.write(question.replace('\n', ' ').replace('\t', ' ').strip() + '\n')
            
            if not all_sample_num or i < all_sample_num:
                qa_pair_wf.write(question.replace('\n', ' ').replace('\t', ' ').strip() + '\t' + answer.replace('\n', ' ').replace('\t', ' ').strip() + '\n')
                qac_triple_wf.write(question.replace('\n', ' ').replace('\t', ' ').strip() + '\t' + answer.replace('\n', ' ').replace('\t', ' ').strip() + '\t' + context + '\n')
                q_corpus_wf.write(question.replace('\n', ' ').replace('\t', ' ').strip() + '\n')

# def extract_trans_from_fake_question(json_file, out_file, test_sample_num, query_answer_path=None):
#     with open(json_file, 'r', encoding='utf-8') as rf, \
#                         open(os.path.join(out_file), 'w', encoding='utf-8') as wf:
#         if query_answer_path:
#             with open(query_answer_path, 'w', encoding='utf-8') as qeury_answer_wf:
#                 for i, json_line in enumerate(rf.readlines()):
#                     line_dict = json.loads(json_line)
#                     if isinstance(line_dict['question'] , list):
#                         question = line_dict['question'][0].replace('习近平', '').replace('习总', '习')
#                     else:
#                         question = line_dict['question'].replace('习近平', '').replace('习总', '习')
#                     answer = line_dict['answer'].replace('习近平', '').replace('习总', '习')
#                     if i < test_sample_num:
#                         qeury_answer_wf.write(question.strip() + '\t' + answer + '\n')
#                         wf.write(question.strip() + '\n')
#                     else:
#                         break
#         else:
#             for i, json_line in enumerate(rf.readlines()):
#                 line_dict = json.loads(json_line)
#                 if isinstance(line_dict['question'] , list):
#                     question = line_dict['question'][0].replace('习近平', '').replace('习总', '习')
#                 else:
#                     question = line_dict['question'].replace('习近平', '').replace('习总', '习')
#                 answer = line_dict['answer'].replace('习近平', '').replace('习总', '习')
#                 if i < test_sample_num:
#                     wf.write(question.strip() + '\n')
#                 else:
#                     break



# def create_test_qq_pair(ori_path=None, trans_path=None, write_path=None):
#     with open(ori_path, 'r', encoding='utf-8') as origin_rf, \
#         open(trans_path, 'r', encoding='utf-8') as trans_rf, \
#         open(write_path, 'w', encoding='utf-8') as wf:
#         for origin, trans in zip(origin_rf, trans_rf):
#             wf.write(trans.strip() + '\t' + origin.strip() + '\n')

if __name__ == '__main__':
    args = parse_args()
    # convert_json_to_data('/root/project/data/faq-qa/yiqing-answer-aware-unsupervised-FAQ-QA/yiqing_data_fake_question.json', '/root/project/data/faq-qa/yiqing-answer-aware-unsupervised-FAQ-QA')
    if args.do_create_faq_corpus:
        convert_json_to_data(args.source_file_path, args.target_dir_path,args.test_sample_num,  args.train_sample_num,  args.all_sample_num)
    # if args.do_extract_trans_from_fake_question:
    #     extract_trans_from_fake_question(args.source_trans_json_file_path, args.target_trans_txt_file_path, args.test_sample_num, query_answer_path=args.query_answer_file_path)
    # if args.do_convert_json_to_data:
    #     create_test_qq_pair(ori_path=args.qq_pair_source_ori_file_path if args.qq_pair_source_ori_file_path else os.path.join(args.target_dir_path, 'dev.csv'), 
    #                         trans_path=args.qq_pair_source_trans_file_path,
    #                         write_path=args.qq_pair_target_file_path,)
                        



            