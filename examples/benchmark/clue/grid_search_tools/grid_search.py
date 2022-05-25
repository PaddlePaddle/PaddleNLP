import time
import subprocess
import random
import sys
from collections import defaultdict

from pynvml import *

nvmlInit()

world_size = nvmlDeviceGetCount()
handles = []
handle_mapping = {}
for i in range(world_size):
    h = nvmlDeviceGetHandleByIndex(i)
    handles.append(h)
    handle_mapping[str(h)] = i


def get_availble(est=15):
    # sort handles according to info.free
    handles.sort(key=lambda x: nvmlDeviceGetMemoryInfo(x).free, reverse=True)
    for i, h in enumerate(handles):
        device_id = handle_mapping[str(h)]
        info = nvmlDeviceGetMemoryInfo(h)
        gb = 1024 * 1024 * 1024
        print(f'- device_id: {device_id}')
        print(f'- free     : {info.free/gb}')
        if info.free / gb >= est:
            return device_id
    return None


# TODO Support multi-machine


def get_mrc_tasks(model_name_or_path):
    learning_rate_list = [1e-5, 2e-5, 3e-5]
    batch_size_list = [24, 32]
    cls_base_grd_acc = 4
    tasks = []
    # chid
    for lr in learning_rate_list:
        for bs in batch_size_list:
            tasks.append(
                f"bash run_chid.sh {model_name_or_path}  {bs} {lr} {cls_base_grd_acc*2}"
            )
    # c3
    for lr in learning_rate_list:
        for bs in batch_size_list:
            tasks.append(
                f"bash run_cmrc.sh {model_name_or_path}  {bs} {lr} {cls_base_grd_acc}"
            )
    # cmrc2018
    for lr in learning_rate_list:
        for bs in batch_size_list:
            grd_acc = bs // 2
            tasks.append(
                f"bash run_chid.sh {model_name_or_path}  {bs} {lr} {grd_acc}")
    return tasks


def get_cls_tasks(model_name_or_path):
    learning_rate_list = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_size_list = [16, 32, 64]
    datasets = [
        'afqmc', 'tnews', 'iflytek', 'ocnli', 'cmnli', 'cluewsc2020', 'csl'
    ]
    # epoch, max_seq_len, grd_acc, dropout
    cls_base_grd_acc = 1
    hyper_params = {
        "afqmc": [[3, 128, cls_base_grd_acc, 0.1]],
        "tnews": [[3, 128, cls_base_grd_acc, 0.1]],
        "iflytek":
        [[3, 128, cls_base_grd_acc, 0.1], [3, 128, cls_base_grd_acc, 0.0]],
        "ocnli": [[5, 128, cls_base_grd_acc, 0.1]],
        "cluewsc2020":
        [[50, 128, cls_base_grd_acc, 0.1], [50, 128, cls_base_grd_acc, 0.0]],
        "csl": [[5, 256, cls_base_grd_acc * 2, 0.1]],
        "cmnli": [[2, 128, cls_base_grd_acc, 0.1]]
    }
    batch_size_8_task = ["cluewsc2020"]
    tasks = []
    for dataset in datasets:
        for lr in learning_rate_list:
            for bs in batch_size_list:
                for hyper_param in hyper_params[dataset]:
                    epoch, max_seq_len, grd_acc, dropout = hyper_param
                    tasks.append(
                        f"bash run_clue_classifier.sh {dataset} {lr} {bs} {epoch} {max_seq_len} {model_name_or_path} {grd_acc} {dropout}"
                    )
    for lr in learning_rate_list:
        for hyper_param in hyper_params["cluewsc2020"]:
            bs = 8
            epoch, max_seq_len, grd_acc, dropout = hyper_param
            tasks.append(
                f"bash run_clue_classifier.sh {dataset} {lr} {bs} {epoch} {max_seq_len} {model_name_or_path} {grd_acc} {dropout}"
            )
    return tasks


def do_task(task):
    tmp = task.split(" ")
    est = 15
    # if int(tmp[4]) * int(tmp[6]) > 32 * 128:
    #     est = 30
    print(est)
    device_id = get_availble(est)
    while device_id is None:
        print("> No device avaliable, wait 15 seconds.")
        time.sleep(15)
        device_id = get_availble()
    task = f"set -x \nexport CUDA_VISIBLE_DEVICES={device_id}\n" + task
    print(f"> Send task \n{task}\n")
    ps = subprocess.Popen(
        task, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return ps


def main():
    # Make sure that dataset has been downloaded first
    status = os.system('python download_dataset.py')
    model_name_or_path = sys.argv[1]
    # """
    tasks = get_cls_tasks(model_name_or_path)
    tasks += get_mrc_tasks(model_name_or_path)

    for x in tasks:
        print(x)
    # sys.exit(0)

    runs = []
    retry = defaultdict(int)
    while len(tasks) > 0 or len(runs) > 0:
        i = 0
        print("\n\n\n>> Round start")
        while i < len(runs):
            returncode = runs[i]["ps"].poll()
            # print(returncode)
            if returncode is not None:
                if returncode != 0:
                    retry[runs[i]["ts"]] += 1
                    print(
                        f"> {runs[i]['ts']} task failed, will retried, tryed {retry[runs[i]['ts']]} times."
                    )
                    output = runs[i]["ps"].communicate()[0]
                    for line in output.decode('utf-8').split("\n"):
                        print(line)
                    if retry[runs[i]["ts"]] <= 5:
                        tasks.append(runs[i]["ts"])
                else:
                    print(f"> Done! {runs[i]['ts']}")
                runs.pop(i)
                i = i - 1
            else:
                print(">> DOING", runs[i]["ts"])
            i += 1

        if len(tasks) > 0:
            task = tasks.pop(0)
            print(f"> Try to append {task}")
            ps = do_task(task)
            runs.append({"ps": ps, "ts": task})

        print(f"> Wait for 10 seconds to start!")
        time.sleep(10)
    # """
    status = os.system('bash extract_acc.sh ' + model_name_or_path)


if __name__ == "__main__":
    main()
