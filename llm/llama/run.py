import argparse
import os
from multiprocessing import Process



def main(args):
  seq_lens = [args.seq_len]
  gpus_list = [args.gpus]

  if args.var_seq_len == "true":
    seq_lens = [pow(2, k) * 2048 for k in range(0, 10)]
    # seq_lens = [pow(2, k) * 2048 for k in range(0, 4)]
    print(f"seq_lens:{seq_lens}")
  if args.var_gpus == "true":
    gpus = [str(i) for i in list(range(8))]
    gpus_list = [",".join(gpus[:n]) for n in [2,4,8]]
    print(f"gpus_list:{gpus_list}")
  for gpus in gpus_list:
    gpu_num = len(gpus.split(","))
    for seq_len in seq_lens:
      cmd = f"bash do_run.sh {args.mode} {seq_len} {gpus} {gpu_num}"
      print(f"execute cmd: {cmd}")
      ret = os.system(cmd)
      if int(ret) != 0:
        print(f"execute cmd:{cmd} fail")
        break
  for gpus in gpus_list:
    gpu_num = len(gpus.split(","))
    for seq_len in seq_lens:
      log_dir = f"{args.mode}_log_seq_{seq_len}_gpus_{gpu_num}"
      if not os.path.exists(log_dir):
        continue
      print(f"analyze logs:{log_dir}")
      # os.system(f"cat {log_dir}/workerlog.0 |grep ' loss:'|tail -1")
      os.system(f"cat {log_dir}/workerlog.0 |grep 'global_step: 10'|tail -1")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", type=str, default="sep", choices=["sep", "tp"])
  parser.add_argument("--seq_len", type=int, default=2048 * pow(2, 3))  # 16k
  parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
  parser.add_argument("--var_seq_len", type=str, default="false", choices=["true", "false"])
  parser.add_argument("--var_gpus", type=str, default="false", choices=["true", "false"])

  args = parser.parse_args()
  print(args)
  main(args)
