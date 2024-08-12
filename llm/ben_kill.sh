ps aux | grep run_finetune | awk '{print $2}'| xargs kill -9
