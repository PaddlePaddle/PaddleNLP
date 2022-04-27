# ## Task: Speech Command Analysis for Audio Expense Claim
#
# Structured information entry is a common application scenario of speech
# command analysis, where we can extract expected keywords from audios in
# an end-to-end way. This technique can economize on manpower and reduce
# error rates.

import sys
import argparse
from paddlenlp import Taskflow
from utils import mandarin_asr_api


def speech_cmd_analysis_pipeline(audio_file, uie_model, schema):
    # automatic speech recognition
    text = mandarin_asr_api(audio_file)

    # extract entities according to schema
    if uie_model is None:
        parser = Taskflow('information_extraction', schema=schema)
    else:
        parser = Taskflow(
            'information_extraction',
            schema=schema,
            task_path=uie_model)
    result = parser(text)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio_file', type=str, required=True, help='The audio file name.')
    parser.add_argument(
        '--uie_model', type=str, default=None, help='The path to uie model.')
    parser.add_argument(
        '--schema', type=list, default=['时间', '出发地', '目的地', '费用'],
        help='The type of entities expected to extract.')
    args = parser.parse_args()

    result = speech_cmd_analysis_pipeline(
        args.audio_file, args.uie_model, args.schema)

    print(result)
    
