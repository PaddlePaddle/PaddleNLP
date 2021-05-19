# 环境
建议 python3.7

# 评估
输入文件格式：
1. 文件以.sql结尾
2. 文件每行的格式："qid\tsql_query\tdb_id",其中predcit文件db_id是可选字段，gold文件db_id是必选字段
3. 评估指标：exact matching score

# 使用

## 命令行

    python text2sql_evaluation.py \
        --g 'data/DuSQL/test_gold.sql' \      # gold文件
        --p 'test_DuSQL.sql' \                # predict文件
        --t 'data/DuSQL/db_schema.json' \     # schema文件
        --d 'DuSQL'                           # 选择dataset（DuSQL、NL2SQL、CSPider可选）

## 接口

    from text2sql_evaluation import evaluate
    score, score_novalue = evaluate('table.json', 'gold.sql', 'pred.sql', dataset='DuSQL')
其中：
    score["all"] = {"exact": exact num, "count": test examples num, "acc": accuracy}
    score_novalue["all"] = {"exact": exact num, "count": test examples num, "acc": accuracy}

## 输出
    with value:
    {"exact": exact correct num, "count": test examples num, "acc": accuracy}
    without value:
    {"exact": exact correct num, "count": test examples num, "acc": accuracy}
其中：
    acc表示最终输出准确率
