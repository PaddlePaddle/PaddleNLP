# Copyright (c) 2021 PaddlePaddle Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm examples for BERT."""

import argparse
import os

from text_formatting.bookcorpus import BookscorpusTextFormatter
from text_formatting.wikicorpus import WikicorpusTextFormatter
from text_sharding import Sharding, EnglishSegmenter, ChineseSegmenter
from utils import create_instances_from_document, write_instance_to_example_file

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--formatted_file", default=None, type=str,
    help="The input train corpus which should be already formatted as one article one line."
    "It can be directory with .txt files or a path to a single file")
parser.add_argument('--skip_formatting', action="store_true", default=True, required=True,
    help="If the input file already have forrmatted as formatted as one article one line, "
    "you can skip text formatting precoess.")
parser.add_argument("--output_dir", default=None, type=str, required=True,
    help="The output directory where the pretrained data will be written.")
parser.add_argument("--model_name", choices=['bert-base-uncased', 'bert-base-chinese', 'bert-wwm-chinese','ernie-1.0'],
    default="bert-base-chinese", required=True,
    help="Select which model to pretrain, defaults to bert-base-chinese.")
parser.add_argument("--max_seq_length", default=128, type=int,
    help="The maximum total input sequence length after WordPiece tokenization. \n"
    "Sequences longer than this will be truncated, and sequences shorter \n"
    "than this will be padded.")
parser.add_argument("--max_word_length", default=4, type=int,
    help="The maximum total chinese characters in a chinese word after chinese word segmentation tokenization.")
parser.add_argument("--dupe_factor", default=10, type=int,
    help="Number of times to duplicate the input data (with different masks).")
parser.add_argument("--max_predictions_per_seq", default=20, type=int, help="Maximum sequence length.")
parser.add_argument("--masked_lm_prob", default=0.15, type=float, help="Masked LM probability.")
parser.add_argument("--short_seq_prob", default=0.1, type=float,
    help="Probability to create a sequence shorter than maximum sequence length")
parser.add_argument("--do_lower_case", action="store_true", default=True,
    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument('--random_seed', type=int, default=10000, help="random seed for initialization")
parser.add_argument('--n_training_shards', type=int, default=256, help='Specify the number of training shards to generate')
parser.add_argument('--n_test_shards', type=int, default=1, help='Specify the number of test shards to generate')
parser.add_argument('--fraction_test_set', type=float, default=0.1,
    help='Specify the fraction (0.1) of the data to withhold for the test data split (based on number of sequences)')
args = parser.parse_args()
# yapf: enable


def create_record_worker(shardingfile_prefix,
                         outputfile_prefix,
                         shard_id,
                         output_format='hdf5'):
    bert_preprocessing_command = 'python ./data/create_pretraining_data.py'
    bert_preprocessing_command += ' --input_file=' + shardingfile_prefix \
        + '_' + str(shard_id) + '.txt'
    bert_preprocessing_command += ' --output_file=' + outputfile_prefix \
        + '_' + str(shard_id) + '.hdf5'
    bert_preprocessing_command += ' --do_lower_case' if args.do_lower_case else ''
    bert_preprocessing_command += ' --max_seq_length=' + str(
        args.max_seq_length)
    bert_preprocessing_command += ' --max_predictions_per_seq=' + str(
        args.max_predictions_per_seq)
    bert_preprocessing_command += ' --masked_lm_prob=' + str(
        args.masked_lm_prob)
    bert_preprocessing_command += ' --random_seed=' + str(args.random_seed)
    bert_preprocessing_command += ' --dupe_factor=' + str(args.dupe_factor)
    bert_preprocessing_command += ' --model_name=' + str(args.model_name)

    bert_preprocessing_process = subprocess.Popen(
        bert_preprocessing_command, shell=True)

    last_process = bert_preprocessing_process

    # This could be better optimized (fine if all take equal time)
    if shard_id % args.n_processes == 0 and shard_id > 0:
        bert_preprocessing_process.wait()
    return last_process


if __name__ == "__main__":
    if not skip_formatting:
        if args.model_name not in [
                "bert-base-uncased", "bert-base-chinese", "bert-wwm-chinese"
        ]:
            logger.error(
                "The implimented text formattting process only fits"
                "bert-base-uncased, bert-base-chinese and bert-wwm-chinese."
                "Preraining model %s you should format the corpus firstly by your own."
            )

        logger.info("=" * 50)
        logger.info("Start to text formatting.")
        if args.model_name == "bert-base-uncased":
            wiki_formatter = WikicorpusTextFormatter('en', args.output_dir)
            formatted_files = [wiki_formatter.formatted_file]

            book_formatter = BookscorpusTextFormatter(args.output_dir)
            formatted_files.append(book_formatter.formatted_file)
        else:
            wiki_formatter = WikicorpusTextFormatter('zh', args.output_dir)
            formatted_files = wiki_formatter.formatted_file

        logger.info("End to text formatting")
    else:
        logger.info("=" * 50)
        logger.info("Skip text formatting, formatted file: %s" %
                    args.formatted_file)
        formatted_files = args.formatted_file

    logger.info("=" * 50)
    logger.info("Start to text Sharding. Formated files: {}".format(
        formatted_files))
    sharding_path = os.path.join(args.output_dir,
        'sharded_training_shards_' + str(args.n_training_shards) \
        + "_test_shards_" + str(args.n_test_shard)) \
        + "_fraction_" + str(args.fraction_test_set)
    os.makedirs(sharding_path)

    # Segmentation is here because all datasets look the same in one article/book/whatever per line format, and
    # it seemed unnecessarily complicated to add an additional preprocessing step to call just for this.
    # For english, we use EnglishSegmenter. For chinese, we use ChineseSegmenter.
    if args.model_name == "bert-base-uncased":
        segmenter = EnglishSegmenter()
    else:
        segmenter = ChineseSegmenter()

    sharding_output_name_prefix = os.path.join(sharding_path, "sharding")
    sharding = TextSharding(formatted_files, sharding_output_name_prefix,
                            args.n_training_shards, args.n_test_shards,
                            args.fraction_test_set)
    sharding.load_articles()
    logging.info("Splitting the articles into sentences.")
    sharding.segment_articles_into_sentences(segmenter)
    sharding.distribute_articles_over_shards()
    sharding.write_shards_to_disk()
    logger.info("End to text sharding. Sharding files save as {}".format(
        sharding_path))

    logger.info("=" * 50)
    logger.info("Start to create pretrainging data and save it to hdf5 files.")
    hdf5_folder = "hdf5_lower_case_" + str(args.do_lower_case) + "_seq_len_" + str(args.max_seq_length) \
        + "_max_pred_" + str(args.max_predictions_per_seq) + "_masked_lm_prob_" + str(args.masked_lm_prob) \
        + "_random_seed_" + str(args.random_seed) + "_dupe_factor_" + str(args.dupe_factor)
    os.makedirs(args.output_dir, hdf5_folder)
    hdf5_folder_prefix = os.path.join(args.output_dir, hdf5_folder, "pretraing")

    for i in range(args.n_training_shards):
        last_process = create_record_worker(
            sharding_output_name_prefix + "_train",
            hdf5_folder_prefix + "_train", i)
    last_process.wait()

    for i in range(args.n_test_shards):
        last_process = create_record_worker(
            sharding_output_name_prefix + '_test', hdf5_folder_prefix + "_test",
            i)
    last_process.wait()

    logger.info("End to create pretrainging data and save it to hdf5 files.")
