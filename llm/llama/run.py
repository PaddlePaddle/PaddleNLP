import os
from multiprocessing import Process


def main():
  seq_lens = [pow(2, k) * 2048 for k in range(4, 10)]
  print(seq_lens)
  for seq_len in seq_lens:
    cmd = f"bash do_run.sh {seq_len}"
    ret = os.system(cmd)
    if int(ret) != 0:
      print(f"execute cmd:{cmd} fail")
      break
  for seq_len in seq_lens:
    log_dir = f"tp_log_{seq_len}"
    if not os.path.exists(log_dir):
      continue
    print(f"sequence length:{seq_len} speed in logs:{log_dir}")
    os.system(f"cat {log_dir}/workerlog.0 |grep ' loss:'|tail -1")

if __name__ == "__main__":
  main()
