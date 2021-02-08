# this file is only used for continuous evaluation test!

import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi

lstm_language_model_static_loss_card1 = CostKpi('lstm_language_model_static_loss_card1', 0.01, 0)
lstm_language_model_static_duration_card1 = DurationKpi(
    'lstm_language_model_static_duration_card1', 0.03, 0, actived=True)
lstm_language_model_padding_loss_card1 = CostKpi('lstm_language_model_padding_loss_card1', 0.01, 0)
lstm_language_model_padding_duration_card1 = DurationKpi(
    'lstm_language_model_padding_duration_card1', 0.03, 0, actived=True)

tracking_kpis = [
    lstm_language_model_static_loss_card1,
    lstm_language_model_static_duration_card1,
    lstm_language_model_padding_loss_card1,
    lstm_language_model_padding_duration_card1,
]


def parse_log(log):
    '''
    This method should be implemented by model developers.

    The suggestion:

    each line in the log should be key, value, for example:

    "
    train_cost\t1.0
    test_cost\t1.0
    train_cost\t1.0
    train_cost\t1.0
    train_acc\t1.2
    "
    '''
    for line in log.split('\n'):
        fs = line.strip().split('\t')
        print(fs)
        if len(fs) == 3 and fs[0] == 'ptblm':
            kpi_name = fs[1]
            kpi_value = float(fs[2])
            yield kpi_name, kpi_value


def log_to_ce(log):
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for (kpi_name, kpi_value) in parse_log(log):
        print(kpi_name, kpi_value)
        kpi_tracker[kpi_name].add_record(kpi_value)
        kpi_tracker[kpi_name].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    log_to_ce(log)
