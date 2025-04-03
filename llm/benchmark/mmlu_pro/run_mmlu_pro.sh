# export http_proxy=http://agent.baidu.com:8891
# export https_proxy=http://agent.baidu.com:8891
# export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com 
IP=127.0.0.1
PORT=9965
python3 evaluate_from_api.py --backend paddle --ip $IP --port $PORT --output_dir ./r1_mla_eval --assigned_subjects business