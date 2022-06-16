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

import argparse
import json
import os
import pickle
import shutil
import sys

import sqlparse

from postprocess_eval import get_candidate_tables


def write_interaction(interaction_list, split, output_dir):
    json_split = os.path.join(output_dir, split + '.json')
    pkl_split = os.path.join(output_dir, split + '.pkl')

    with open(json_split, 'w') as outfile:
        for interaction in interaction_list:
            json.dump(interaction, outfile, indent=4)
            outfile.write('\n')

    new_objs = []
    for i, obj in enumerate(interaction_list):
        new_interaction = []
        for ut in obj["interaction"]:
            sql = ut["sql"]
            sqls = [sql]
            tok_sql_list = []
            for sql in sqls:
                results = []
                tokenized_sql = sql.split()
                tok_sql_list.append((tokenized_sql, results))
            ut["sql"] = tok_sql_list
            new_interaction.append(ut)
        obj["interaction"] = new_interaction
        new_objs.append(obj)

    with open(pkl_split, 'wb') as outfile:
        pickle.dump(new_objs, outfile)

    return


def read_database_schema(database_schema_filename, schema_tokens, column_names,
                         database_schemas_dict):
    with open(database_schema_filename) as f:
        database_schemas = json.load(f)

    def get_schema_tokens(table_schema):
        column_names_surface_form = []
        column_names = []
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                table_name = table_names_original[table_id]
                column_name_surface_form = '{}.{}'.format(
                    table_name, column_name)
            else:
                # this is just *
                column_name_surface_form = column_name
            column_names_surface_form.append(column_name_surface_form.lower())
            column_names.append(column_name.lower())

        # also add table_name.*
        for table_name in table_names_original:
            column_names_surface_form.append('{}.*'.format(table_name.lower()))

        return column_names_surface_form, column_names

    for table_schema in database_schemas:
        database_id = table_schema['db_id']
        database_schemas_dict[database_id] = table_schema
        schema_tokens[database_id], column_names[
            database_id] = get_schema_tokens(table_schema)

    return schema_tokens, column_names, database_schemas_dict


def remove_from_with_join(format_sql_2):
    used_tables_list = []
    format_sql_3 = []
    table_to_name = {}
    table_list = []
    old_table_to_name = {}
    old_table_list = []
    for sub_sql in format_sql_2.split('\n'):
        if 'select ' in sub_sql:
            # only replace alias: t1 -> table_name, t2 -> table_name, etc...
            if len(table_list) > 0:
                for i in range(len(format_sql_3)):
                    for table, name in table_to_name.items():
                        format_sql_3[i] = format_sql_3[i].replace(table, name)

            old_table_list = table_list
            old_table_to_name = table_to_name
            table_to_name = {}
            table_list = []
            format_sql_3.append(sub_sql)
        elif sub_sql.startswith('from'):
            new_sub_sql = None
            sub_sql_tokens = sub_sql.split()
            for t_i, t in enumerate(sub_sql_tokens):
                if t == 'as':
                    table_to_name[sub_sql_tokens[t_i +
                                                 1]] = sub_sql_tokens[t_i - 1]
                    table_list.append(sub_sql_tokens[t_i - 1])
                elif t == ')' and new_sub_sql is None:
                    # new_sub_sql keeps some trailing parts after ')'
                    new_sub_sql = ' '.join(sub_sql_tokens[t_i:])
            if len(table_list) > 0:
                # if it's a from clause with join
                if new_sub_sql is not None:
                    format_sql_3.append(new_sub_sql)

                used_tables_list.append(table_list)
            else:
                # if it's a from clause without join
                table_list = old_table_list
                table_to_name = old_table_to_name
                assert 'join' not in sub_sql
                if new_sub_sql is not None:
                    sub_sub_sql = sub_sql[:-len(new_sub_sql)].strip()
                    assert len(sub_sub_sql.split()) == 2
                    used_tables_list.append([sub_sub_sql.split()[1]])
                    format_sql_3.append(sub_sub_sql)
                    format_sql_3.append(new_sub_sql)
                elif 'join' not in sub_sql:
                    assert len(sub_sql.split()) == 2 or len(
                        sub_sql.split()) == 1
                    if len(sub_sql.split()) == 2:
                        used_tables_list.append([sub_sql.split()[1]])

                    format_sql_3.append(sub_sql)
                else:
                    print('bad from clause in remove_from_with_join')
                    exit()
        else:
            format_sql_3.append(sub_sql)

    if len(table_list) > 0:
        for i in range(len(format_sql_3)):
            for table, name in table_to_name.items():
                format_sql_3[i] = format_sql_3[i].replace(table, name)

    used_tables = []
    for t in used_tables_list:
        for tt in t:
            used_tables.append(tt)
    used_tables = list(set(used_tables))

    return format_sql_3, used_tables, used_tables_list


def remove_from_without_join(format_sql_3, column_names, schema_tokens):
    format_sql_4 = []
    table_name = None
    for sub_sql in format_sql_3.split('\n'):
        if 'select ' in sub_sql:
            if table_name:
                for i in range(len(format_sql_4)):
                    tokens = format_sql_4[i].split()
                    for ii, token in enumerate(tokens):
                        if token in column_names and tokens[ii - 1] != '.':
                            if (ii + 1 < len(tokens) and tokens[ii + 1] != '.'
                                    and tokens[ii + 1] != '('
                                ) or ii + 1 == len(tokens):
                                if '{}.{}'.format(table_name,
                                                  token) in schema_tokens:
                                    tokens[ii] = '{} . {}'.format(
                                        table_name, token)
                    format_sql_4[i] = ' '.join(tokens)

            format_sql_4.append(sub_sql)
        elif sub_sql.startswith('from'):
            sub_sql_tokens = sub_sql.split()
            if len(sub_sql_tokens) == 1:
                table_name = None
            elif len(sub_sql_tokens) == 2:
                table_name = sub_sql_tokens[1]
            else:
                print('bad from clause in remove_from_without_join')
                print(format_sql_3)
                exit()
        else:
            format_sql_4.append(sub_sql)

    if table_name:
        for i in range(len(format_sql_4)):
            tokens = format_sql_4[i].split()
            for ii, token in enumerate(tokens):
                if token in column_names and tokens[ii - 1] != '.':
                    if (ii + 1 < len(tokens) and tokens[ii + 1] != '.'
                            and tokens[ii + 1] != '(') or ii + 1 == len(tokens):
                        if '{}.{}'.format(table_name, token) in schema_tokens:
                            tokens[ii] = '{} . {}'.format(table_name, token)
            format_sql_4[i] = ' '.join(tokens)

    return format_sql_4


def add_table_name(format_sql_3, used_tables, column_names, schema_tokens):
    # If just one table used, easy case, replace all column_name -> table_name.column_name
    if len(used_tables) == 1:
        table_name = used_tables[0]
        format_sql_4 = []
        for sub_sql in format_sql_3.split('\n'):
            if sub_sql.startswith('from'):
                format_sql_4.append(sub_sql)
                continue

            tokens = sub_sql.split()
            for ii, token in enumerate(tokens):
                if token in column_names and tokens[ii - 1] != '.':
                    if (ii + 1 < len(tokens) and tokens[ii + 1] != '.'
                            and tokens[ii + 1] != '(') or ii + 1 == len(tokens):
                        if '{}.{}'.format(table_name, token) in schema_tokens:
                            tokens[ii] = '{} . {}'.format(table_name, token)
            format_sql_4.append(' '.join(tokens))
        return format_sql_4

    def get_table_name_for(token):
        table_names = []
        for table_name in used_tables:
            if '{}.{}'.format(table_name, token) in schema_tokens:
                table_names.append(table_name)
        if len(table_names) == 0:
            return 'table'
        if len(table_names) > 1:
            return None
        else:
            return table_names[0]

    format_sql_4 = []
    for sub_sql in format_sql_3.split('\n'):
        if sub_sql.startswith('from'):
            format_sql_4.append(sub_sql)
            continue

        tokens = sub_sql.split()
        for ii, token in enumerate(tokens):
            # skip *
            if token == '*':
                continue
            if token in column_names and tokens[ii - 1] != '.':
                if (ii + 1 < len(tokens) and tokens[ii + 1] != '.'
                        and tokens[ii + 1] != '(') or ii + 1 == len(tokens):
                    table_name = get_table_name_for(token)
                    if table_name:
                        tokens[ii] = '{} . {}'.format(table_name, token)
        format_sql_4.append(' '.join(tokens))

    return format_sql_4


def check_oov(format_sql_final, output_vocab, schema_tokens):
    for sql_tok in format_sql_final.split():
        if not (sql_tok in schema_tokens or sql_tok in output_vocab):
            print('OOV!', sql_tok)
            raise Exception('OOV')


def normalize_space(format_sql):
    format_sql_1 = [
        ' '.join(sub_sql.strip().replace(',', ' , ').replace(
            '.', ' . ').replace('(', ' ( ').replace(')', ' ) ').split())
        for sub_sql in format_sql.split('\n')
    ]
    format_sql_1 = '\n'.join(format_sql_1)

    format_sql_2 = format_sql_1.replace('\njoin', ' join').replace(
        ',\n', ', ').replace(' where', '\nwhere').replace(
            ' intersect',
            '\nintersect').replace('\nand',
                                   ' and').replace('order by t2 .\nstart desc',
                                                   'order by t2 . start desc')

    format_sql_2 = format_sql_2.replace(
        'select\noperator', 'select operator'
    ).replace('select\nconstructor', 'select constructor').replace(
        'select\nstart',
        'select start').replace('select\ndrop', 'select drop').replace(
            'select\nwork',
            'select work').replace('select\ngroup', 'select group').replace(
                'select\nwhere_built', 'select where_built'
            ).replace('select\norder', 'select order').replace(
                'from\noperator', 'from operator').replace(
                    'from\nforward',
                    'from forward').replace('from\nfor', 'from for').replace(
                        'from\ndrop', 'from drop').replace(
                            'from\norder', 'from order').replace(
                                '.\nstart', '. start').replace(
                                    '.\norder', '. order').replace(
                                        '.\noperator', '. operator'
                                    ).replace('.\nsets', '. sets').replace(
                                        '.\nwhere_built',
                                        '. where_built').replace(
                                            '.\nwork', '. work').replace(
                                                '.\nconstructor',
                                                '. constructor').replace(
                                                    '.\ngroup',
                                                    '. group').replace(
                                                        '.\nfor',
                                                        '. for').replace(
                                                            '.\ndrop',
                                                            '. drop').replace(
                                                                '.\nwhere',
                                                                '. where')

    format_sql_2 = format_sql_2.replace('group by', 'group_by').replace(
        'order by',
        'order_by').replace('! =', '!=').replace('limit value', 'limit_value')
    return format_sql_2


def normalize_final_sql(format_sql_5):
    format_sql_final = format_sql_5.replace('\n', ' ').replace(
        ' . ', '.').replace('group by',
                            'group_by').replace('order by', 'order_by').replace(
                                '! =', '!=').replace('limit value',
                                                     'limit_value')

    # normalize two bad sqls
    if 't1' in format_sql_final or 't2' in format_sql_final or 't3' in format_sql_final or 't4' in format_sql_final:
        format_sql_final = format_sql_final.replace('t2.dormid', 'dorm.dormid')

    # This is the failure case of remove_from_without_join()
    format_sql_final = format_sql_final.replace(
        'select city.city_name where city.state_name in ( select state.state_name where state.state_name in ( select river.traverse where river.river_name = value ) and state.area = ( select min ( state.area ) where state.state_name in ( select river.traverse where river.river_name = value ) ) ) order_by population desc limit_value',
        'select city.city_name where city.state_name in ( select state.state_name where state.state_name in ( select river.traverse where river.river_name = value ) and state.area = ( select min ( state.area ) where state.state_name in ( select river.traverse where river.river_name = value ) ) ) order_by city.population desc limit_value'
    )

    return format_sql_final


def parse_sql(sql_string, db_id, column_names, output_vocab, schema_tokens,
              schema):
    format_sql = sqlparse.format(sql_string, reindent=True)
    format_sql_2 = normalize_space(format_sql)

    num_from = sum([
        1 for sub_sql in format_sql_2.split('\n') if sub_sql.startswith('from')
    ])
    num_select = format_sql_2.count('select ') + format_sql_2.count('select\n')

    format_sql_3, used_tables, used_tables_list = remove_from_with_join(
        format_sql_2)

    format_sql_3 = '\n'.join(format_sql_3)
    format_sql_4 = add_table_name(format_sql_3, used_tables, column_names,
                                  schema_tokens)

    format_sql_4 = '\n'.join(format_sql_4)
    format_sql_5 = remove_from_without_join(format_sql_4, column_names,
                                            schema_tokens)

    format_sql_5 = '\n'.join(format_sql_5)
    format_sql_final = normalize_final_sql(format_sql_5)

    candidate_tables_id, table_names_original = get_candidate_tables(
        format_sql_final, schema)

    failure = False
    if len(candidate_tables_id) != len(used_tables):
        failure = True

    check_oov(format_sql_final, output_vocab, schema_tokens)

    return format_sql_final


def read_spider_split(split_json, interaction_list, database_schemas,
                      column_names, output_vocab, schema_tokens, remove_from):
    with open(split_json) as f:
        split_data = json.load(f)
    print('read_spider_split', split_json, len(split_data))

    for i, ex in enumerate(split_data):
        db_id = ex['db_id']

        final_sql = []
        skip = False
        for query_tok in ex['query_toks_no_value']:
            if query_tok != '.' and '.' in query_tok:
                # invalid sql; didn't use table alias in join
                final_sql += query_tok.replace('.', ' . ').split()
                skip = True
            else:
                final_sql.append(query_tok)
        final_sql = ' '.join(final_sql)

        if skip and 'train' in split_json:
            continue

        if remove_from:
            final_sql_parse = parse_sql(final_sql, db_id, column_names[db_id],
                                        output_vocab, schema_tokens[db_id],
                                        database_schemas[db_id])
        else:
            final_sql_parse = final_sql

        final_utterance = ' '.join(ex['question_toks'])

        if db_id not in interaction_list:
            interaction_list[db_id] = []

        interaction = {}
        interaction['id'] = ''
        interaction['scenario'] = ''
        interaction['database_id'] = db_id
        interaction['interaction_id'] = len(interaction_list[db_id])
        interaction['final'] = {}
        interaction['final']['utterance'] = final_utterance
        interaction['final']['sql'] = final_sql_parse
        interaction['interaction'] = [{
            'utterance': final_utterance,
            'sql': final_sql_parse
        }]
        interaction_list[db_id].append(interaction)

    return interaction_list


def read_data_json(split_json, interaction_list, database_schemas, column_names,
                   output_vocab, schema_tokens, remove_from):
    with open(split_json) as f:
        split_data = json.load(f)
    print('read_data_json', split_json, len(split_data))

    for interaction_data in split_data:
        db_id = interaction_data['database_id']
        final_sql = interaction_data['final']['query']
        final_utterance = interaction_data['final']['utterance']

        if db_id not in interaction_list:
            interaction_list[db_id] = []

        # no interaction_id in train
        if 'interaction_id' in interaction_data['interaction']:
            interaction_id = interaction_data['interaction']['interaction_id']
        else:
            interaction_id = len(interaction_list[db_id])

        interaction = {}
        interaction['id'] = ''
        interaction['scenario'] = ''
        interaction['database_id'] = db_id
        interaction['interaction_id'] = interaction_id
        interaction['final'] = {}
        interaction['final']['utterance'] = final_utterance
        interaction['final']['sql'] = final_sql
        interaction['interaction'] = []

        for turn in interaction_data['interaction']:
            turn_sql = []
            skip = False
            for query_tok in turn['query_toks_no_value']:
                if query_tok != '.' and '.' in query_tok:
                    # invalid sql; didn't use table alias in join
                    turn_sql += query_tok.replace('.', ' . ').split()
                    skip = True
                else:
                    turn_sql.append(query_tok)
            turn_sql = ' '.join(turn_sql)

            # Correct some human sql annotation error
            turn_sql = turn_sql.replace(
                'select f_id from files as t1 join song as t2 on t1 . f_id = t2 . f_id',
                'select t1 . f_id from files as t1 join song as t2 on t1 . f_id = t2 . f_id'
            )
            turn_sql = turn_sql.replace('select name from climber mountain',
                                        'select name from climber')
            turn_sql = turn_sql.replace(
                'select sid from sailors as t1 join reserves as t2 on t1 . sid = t2 . sid join boats as t3 on t3 . bid = t2 . bid',
                'select t1 . sid from sailors as t1 join reserves as t2 on t1 . sid = t2 . sid join boats as t3 on t3 . bid = t2 . bid'
            )
            turn_sql = turn_sql.replace('select avg ( price ) from goods )',
                                        'select avg ( price ) from goods')
            turn_sql = turn_sql.replace(
                'select min ( annual_fuel_cost ) , from vehicles',
                'select min ( annual_fuel_cost ) from vehicles')
            turn_sql = turn_sql.replace(
                'select * from goods where price < ( select avg ( price ) from goods',
                'select * from goods where price < ( select avg ( price ) from goods )'
            )
            turn_sql = turn_sql.replace(
                'select distinct id , price from goods where price < ( select avg ( price ) from goods',
                'select distinct id , price from goods where price < ( select avg ( price ) from goods )'
            )
            turn_sql = turn_sql.replace(
                'select id from goods where price > ( select avg ( price ) from goods',
                'select id from goods where price > ( select avg ( price ) from goods )'
            )

            if skip and 'train' in split_json:
                continue

            if remove_from:
                try:
                    turn_sql_parse = parse_sql(turn_sql, db_id,
                                               column_names[db_id],
                                               output_vocab,
                                               schema_tokens[db_id],
                                               database_schemas[db_id])
                except:
                    print('continue')
                    continue
            else:
                turn_sql_parse = turn_sql

            if 'utterance_toks' in turn:
                turn_utterance = ' '.join(turn['utterance_toks'])  # not lower()
            else:
                turn_utterance = turn['utterance']

            interaction['interaction'].append({
                'utterance': turn_utterance,
                'sql': turn_sql_parse
            })
        if len(interaction['interaction']) > 0:
            interaction_list[db_id].append(interaction)

    return interaction_list


def read_spider(spider_dir, database_schemas, column_names, output_vocab,
                schema_tokens, remove_from):
    interaction_list = {}

    train_json = os.path.join(spider_dir, 'train.json')
    interaction_list = read_spider_split(train_json, interaction_list,
                                         database_schemas, column_names,
                                         output_vocab, schema_tokens,
                                         remove_from)

    dev_json = os.path.join(spider_dir, 'dev.json')
    interaction_list = read_spider_split(dev_json, interaction_list,
                                         database_schemas, column_names,
                                         output_vocab, schema_tokens,
                                         remove_from)

    return interaction_list


def read_sparc(sparc_dir, database_schemas, column_names, output_vocab,
               schema_tokens, remove_from):
    interaction_list = {}

    train_json = os.path.join(sparc_dir, 'train_no_value.json')
    interaction_list = read_data_json(train_json, interaction_list,
                                      database_schemas, column_names,
                                      output_vocab, schema_tokens, remove_from)

    dev_json = os.path.join(sparc_dir, 'dev_no_value.json')
    interaction_list = read_data_json(dev_json, interaction_list,
                                      database_schemas, column_names,
                                      output_vocab, schema_tokens, remove_from)

    return interaction_list


def read_cosql(cosql_dir, database_schemas, column_names, output_vocab,
               schema_tokens, remove_from):
    interaction_list = {}

    train_json = os.path.join(cosql_dir, 'train.json')
    interaction_list = read_data_json(train_json, interaction_list,
                                      database_schemas, column_names,
                                      output_vocab, schema_tokens, remove_from)

    dev_json = os.path.join(cosql_dir, 'dev.json')
    interaction_list = read_data_json(dev_json, interaction_list,
                                      database_schemas, column_names,
                                      output_vocab, schema_tokens, remove_from)

    return interaction_list


def read_db_split(data_dir):
    train_database = []
    with open(os.path.join(data_dir, 'train_db_ids.txt')) as f:
        for line in f:
            train_database.append(line.strip())

    dev_database = []
    with open(os.path.join(data_dir, 'dev_db_ids.txt')) as f:
        for line in f:
            dev_database.append(line.strip())

    return train_database, dev_database


def preprocess(dataset, remove_from=False):
    # Validate output_vocab
    output_vocab = [
        '_UNK', '_EOS', '.', 't1', 't2', '=', 'select', 'from', 'as', 'value',
        'join', 'on', ')', '(', 'where', 't3', 'by', ',', 'count', 'group',
        'order', 'distinct', 't4', 'and', 'limit', 'desc', '>', 'avg', 'having',
        'max', 'in', '<', 'sum', 't5', 'intersect', 'not', 'min', 'except',
        'or', 'asc', 'like', '!', 'union', 'between', 't6', '-', 't7', '+', '/'
    ]
    if remove_from:
        output_vocab = [
            '_UNK', '_EOS', '=', 'select', 'value', ')', '(', 'where', ',',
            'count', 'group_by', 'order_by', 'distinct', 'and', 'limit_value',
            'limit', 'desc', '>', 'avg', 'having', 'max', 'in', '<', 'sum',
            'intersect', 'not', 'min', 'except', 'or', 'asc', 'like', '!=',
            'union', 'between', '-', '+', '/'
        ]
    print('size of output_vocab', len(output_vocab))
    print('output_vocab', output_vocab)
    print()

    if dataset == 'spider':
        spider_dir = 'data/spider/'
        database_schema_filename = 'data/spider/tables.json'
        output_dir = 'data/spider_data'
        if remove_from:
            output_dir = 'data/spider_data_removefrom'
        train_database, dev_database = read_db_split(spider_dir)
    elif dataset == 'sparc':
        sparc_dir = 'data/sparc/'
        database_schema_filename = 'data/sparc/tables.json'
        output_dir = 'data/sparc_data'
        if remove_from:
            output_dir = 'data/sparc_data_removefrom'
        train_database, dev_database = read_db_split(sparc_dir)
    elif dataset == 'cosql':
        cosql_dir = 'data/cosql/'
        database_schema_filename = 'data/cosql/tables.json'
        output_dir = 'data/cosql_data'
        if remove_from:
            output_dir = 'data/cosql_data_removefrom'
        train_database, dev_database = read_db_split(cosql_dir)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    schema_tokens = {}
    column_names = {}
    database_schemas = {}

    print('Reading spider database schema file')
    schema_tokens, column_names, database_schemas = read_database_schema(
        database_schema_filename, schema_tokens, column_names, database_schemas)
    num_database = len(schema_tokens)
    print('num_database', num_database, len(train_database), len(dev_database))
    print('total number of schema_tokens / databases:', len(schema_tokens))

    output_database_schema_filename = os.path.join(output_dir, 'tables.json')
    with open(output_database_schema_filename, 'w') as outfile:
        json.dump([v for k, v in database_schemas.items()], outfile, indent=4)

    if dataset == 'spider':
        interaction_list = read_spider(spider_dir, database_schemas,
                                       column_names, output_vocab,
                                       schema_tokens, remove_from)
    elif dataset == 'sparc':
        interaction_list = read_sparc(sparc_dir, database_schemas, column_names,
                                      output_vocab, schema_tokens, remove_from)
    elif dataset == 'cosql':
        interaction_list = read_cosql(cosql_dir, database_schemas, column_names,
                                      output_vocab, schema_tokens, remove_from)

    print('interaction_list length', len(interaction_list))

    train_interaction = []
    for database_id in interaction_list:
        if database_id not in dev_database:
            train_interaction += interaction_list[database_id]

    dev_interaction = []
    for database_id in dev_database:
        if database_id in interaction_list.keys():
            dev_interaction += interaction_list[database_id]

    print('train interaction: ', len(train_interaction))
    print('dev interaction: ', len(dev_interaction))

    write_interaction(train_interaction, 'train', output_dir)
    write_interaction(dev_interaction, 'dev', output_dir)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=('spider', 'sparc', 'cosql'),
                        default='sparc')
    parser.add_argument('--remove_from', action='store_true', default=False)
    args = parser.parse_args()
    preprocess(args.dataset, args.remove_from)
