#setup custom tpc library
unset FLAGS_selected_intel_hpus
export GC_KERNEL_PATH=/workspace/pdpd_automation/repo/PaddleCustomDevice/backends/intel_hpu/build/libcustom_tpc_perf_lib.so:/usr/lib/habanalabs/libtpc_kernels.so

#test cmdline example
#PR test cases
python e2e-test-run.py --context pr --data /data/ckpt/ --filter stable --device intel_hpu --junit test_result.xml --platform gaudi2d
#BAT test cases
python e2e-test-run.py --context bat --data /data/ckpt/ --filter stable --device intel_hpu --junit test_result.xml --platform gaudi2d
#smoke test cases
python e2e-test-run.py --context sanity --data /data/ckpt/ --filter stable --device intel_hpu --junit test_result.xml --platform gaudi2d
python e2e-test-run.py --context sanity --data /data/ckpt/ --filter stable --device intel_hpu:2 --junit test_result.xml --platform gaudi2d

#static graph mode Inference
export PYTHONPATH=$PYTHONPATH:/workspace/pdpd_automation/repo/PaddleNLP/
export FLAGS_intel_hpu_execution_queue_size=10
#export relative static mode file
python export_model.py --model_name_or_path /data/ckpt/meta-llama/Llama-2-7b-chat/ --inference_model --output_path ./inference --dtype bfloat16 --device intel_hpu
#run static mode inference
python e2e-test-run.py --context sanity --data /data/ckpt/ --filter stable --device intel_hpu:2 --mode static --junit test_result.xml --platform gaudi2d
