#!/bin/bash
cd /root/PaddleNLP/applications/experimental/pipelines/
sh build_index.sh
nohup sh run_server.sh > server.log 2>&1 &
sleep 10
lsof -i:8899
nohup sh run_client.sh > client.log 2>&1 &
