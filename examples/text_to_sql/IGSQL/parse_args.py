#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
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

import sys

args = sys.argv

import os
import argparse


def interpret_args():
    """ Interprets the command line arguments, and returns a dictionary. """
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_gpus", type=bool, default=1)

    ### Data parameters
    parser.add_argument(
        '--raw_train_filename',
        type=str,
        default='../atis_data/data/resplit/processed/train_with_tables.pkl')
    parser.add_argument(
        '--raw_dev_filename',
        type=str,
        default='../atis_data/data/resplit/processed/dev_with_tables.pkl')
    parser.add_argument(
        '--raw_validation_filename',
        type=str,
        default='../atis_data/data/resplit/processed/valid_with_tables.pkl')
    parser.add_argument(
        '--raw_test_filename',
        type=str,
        default='../atis_data/data/resplit/processed/test_with_tables.pkl')

    parser.add_argument('--data_directory', type=str, default='processed_data')

    parser.add_argument('--processed_train_filename',
                        type=str,
                        default='train.pkl')
    parser.add_argument('--processed_dev_filename', type=str, default='dev.pkl')
    parser.add_argument('--processed_validation_filename',
                        type=str,
                        default='validation.pkl')
    parser.add_argument('--processed_test_filename',
                        type=str,
                        default='test.pkl')

    parser.add_argument('--database_schema_filename', type=str, default=None)
    parser.add_argument('--embedding_filename', type=str, default=None)

    parser.add_argument('--input_vocabulary_filename',
                        type=str,
                        default='input_vocabulary.pkl')
    parser.add_argument('--output_vocabulary_filename',
                        type=str,
                        default='output_vocabulary.pkl')

    parser.add_argument('--input_key', type=str, default='utterance')

    parser.add_argument('--anonymize', type=bool, default=False)
    parser.add_argument('--anonymization_scoring', type=bool, default=False)
    parser.add_argument('--use_snippets', type=bool, default=False)

    parser.add_argument('--use_previous_query', type=bool, default=True)
    parser.add_argument('--maximum_queries', type=int, default=1)
    parser.add_argument('--use_copy_switch', type=bool, default=False)
    parser.add_argument('--use_query_attention', type=bool, default=True)

    parser.add_argument('--use_utterance_attention', type=bool, default=True)

    parser.add_argument('--scheduler', type=bool, default=False)

    parser.add_argument('--use_bert', type=bool, default=True)
    parser.add_argument("--bert_input_version", type=str, default='v1')
    parser.add_argument('--fine_tune_bert', type=bool, default=True)
    parser.add_argument('--lr_bert',
                        default=1e-5,
                        type=float,
                        help='BERT model learning rate.')

    ### Debugging/logging parameters
    parser.add_argument('--reload_embedding', type=bool, default=False)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--deterministic', type=bool, default=False)
    parser.add_argument('--num_train', type=int, default=-1)

    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--results_file', type=str, default='results.txt')

    ### Model architecture
    parser.add_argument('--input_embedding_size', type=int, default=300)
    parser.add_argument('--output_embedding_size', type=int, default=300)

    parser.add_argument('--encoder_state_size', type=int, default=300)
    parser.add_argument('--decoder_state_size', type=int, default=300)

    parser.add_argument('--encoder_num_layers', type=int, default=1)
    parser.add_argument('--decoder_num_layers', type=int, default=1)
    parser.add_argument('--snippet_num_layers', type=int, default=1)

    parser.add_argument('--maximum_utterances', type=int, default=5)
    parser.add_argument('--state_positional_embeddings',
                        type=bool,
                        default=True)
    parser.add_argument('--positional_embedding_size', type=int, default=50)

    parser.add_argument('--snippet_age_embedding', type=bool, default=False)
    parser.add_argument('--snippet_age_embedding_size', type=int, default=64)
    parser.add_argument('--max_snippet_age_embedding', type=int, default=4)
    parser.add_argument('--previous_decoder_snippet_encoding',
                        type=bool,
                        default=False)

    parser.add_argument('--discourse_level_lstm', type=bool, default=True)

    parser.add_argument('--use_schema_attention', type=bool, default=True)
    parser.add_argument('--use_encoder_attention', type=bool, default=True)

    parser.add_argument('--use_schema_encoder', type=bool, default=True)
    parser.add_argument('--use_schema_self_attention', type=bool, default=False)
    parser.add_argument('--use_schema_encoder_2', type=bool, default=False)

    ### Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_maximum_sql_length', type=int,
                        default=400)  #200
    parser.add_argument('--train_evaluation_size', type=int, default=100)

    parser.add_argument('--dropout_amount', type=float, default=0.5)

    parser.add_argument('--initial_patience', type=float, default=10.)
    parser.add_argument('--patience_ratio', type=float, default=1.01)

    parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_ratio', type=float, default=0.9)

    parser.add_argument('--interaction_level', type=bool, default=True)
    parser.add_argument('--reweight_batch', type=bool, default=True)
    parser.add_argument('--gnn_layer_number', type=int, default=1)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--warmup_step', type=int, default=1000)

    ### Setting
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--save_file', type=str, default="")
    parser.add_argument('--enable_testing', type=bool, default=False)
    parser.add_argument('--use_predicted_queries', type=bool, default=False)
    parser.add_argument('--evaluate_split', type=str, default='valid')
    parser.add_argument('--evaluate_with_gold_forcing',
                        type=bool,
                        default=False)
    parser.add_argument('--eval_maximum_sql_length', type=int, default=400)
    parser.add_argument('--results_note', type=str, default='')
    parser.add_argument('--compute_metrics', type=bool, default=False)

    parser.add_argument('--reference_results', type=str, default='')

    parser.add_argument('--interactive', type=bool, default=False)

    parser.add_argument('--database_username', type=str, default="aviarmy")
    parser.add_argument('--database_password', type=str, default="aviarmy")
    parser.add_argument('--database_timeout', type=int, default=2)

    parser.add_argument('--all_in_one_trainer', type=bool, default=False)

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if not (args.train or args.evaluate or args.interactive or args.attention):
        raise ValueError('You need to be training or evaluating')
    if args.enable_testing and not args.evaluate:
        raise ValueError('You should evaluate the model if enabling testing')

    return args
