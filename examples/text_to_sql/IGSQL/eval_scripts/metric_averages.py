import json
import sys

predictions = [
    json.loads(line) for line in open(sys.argv[1]).readlines() if line
]

string_count = 0.
sem_count = 0.
syn_count = 0.
table_count = 0.
strict_table_count = 0.

precision_denom = 0.
precision = 0.
recall_denom = 0.
recall = 0.
f1_score = 0.
f1_denom = 0.

time = 0.

for prediction in predictions:
    if prediction["correct_string"]:
        string_count += 1.
    if prediction["semantic"]:
        sem_count += 1.
    if prediction["syntactic"]:
        syn_count += 1.
    if prediction["correct_table"]:
        table_count += 1.
    if prediction["strict_correct_table"]:
        strict_table_count += 1.
    if prediction["gold_tables"] != "[[]]":
        precision += prediction["table_prec"]
        precision_denom += 1
    if prediction["pred_table"] != "[]":
        recall += prediction["table_rec"]
        recall_denom += 1

        if prediction["gold_tables"] != "[[]]":
            f1_score += prediction["table_f1"]
            f1_denom += 1

num_p = len(predictions)
print("string precision: " + str(string_count / num_p))
print("% semantic: " + str(sem_count / num_p))
print("% syntactic: " + str(syn_count / num_p))
print("table prec: " + str(table_count / num_p))
print("strict table prec: " + str(strict_table_count / num_p))
print("table row prec: " + str(precision / precision_denom))
print("table row recall: " + str(recall / recall_denom))
print("table row f1: " + str(f1_score / f1_denom))
print("inference time: " + str(time / num_p))
