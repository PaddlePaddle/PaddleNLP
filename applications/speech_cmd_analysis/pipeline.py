# ## Task: Speech Command Analysis for Audio Expense Claim
#
# Structured information entry is a common application scenario of speech
# command analysis, where we can extract expected keywords from audios in
# an end-to-end way. This technique can economize on manpower and reduce
# error rates.

import os
import sys
import json
import argparse
import pprint
from tqdm import tqdm
from paddlenlp import Taskflow
from utils import mandarin_asr_api

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file',
                        type=str,
                        required=True,
                        help='The audio file name.')
    parser.add_argument('--api_key',
                        type=str,
                        required=True,
                        help='The app key applied on Baidu AI Platform.')
    parser.add_argument(
        '--secret_key',
        type=str,
        required=True,
        help='The app secret key generated on Baidu AI Platform.')
    parser.add_argument('--uie_model',
                        type=str,
                        default=None,
                        help='The path to uie model.')
    parser.add_argument('--schema',
                        type=str,
                        nargs='+',
                        default=['时间', '出发地', '目的地', '费用'],
                        help='The type of entities expected to extract.')
    parser.add_argument(
        '--save_file',
        type=str,
        default='./uie_results.txt',
        help='The path to save the recognised text and schemas.')
    args = parser.parse_args()

    if os.path.isfile(args.audio_file):
        audios = [args.audio_file]
    elif os.path.isdir(args.audio_file):
        audios = [x for x in os.listdir(args.audio_file)]
        audios = [os.path.join(args.audio_file, x) for x in audios]
    else:
        raise Exception('%s is neither valid path nor file!' % args.audio_file)

    audios = [x for x in audios if x.endswith('.wav')]
    if len(audios) == 0:
        raise Exception('No valid .wav file! Please check %s.' %
                        args.audio_file)

    if args.uie_model is None:
        parser = Taskflow('information_extraction', schema=args.schema)
    else:
        parser = Taskflow('information_extraction',
                          schema=args.schema,
                          task_path=args.uie_model)

    with open(args.save_file, 'w') as fp:
        for audio_file in tqdm(audios):
            # automatic speech recognition
            text = mandarin_asr_api(args.api_key, args.secret_key, audio_file)
            # extract entities according to schema
            result = parser(text)
            fp.write(text + '\n')
            fp.write(json.dumps(result, ensure_ascii=False) + '\n\n')
            print(text)
            pprint.pprint(result)
