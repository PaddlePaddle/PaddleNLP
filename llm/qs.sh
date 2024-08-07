#python test_faiss_gpu.py  --checkpoint_path checkpoints/ckpt_quant/checkpoint-60 --output ckpt --quant_bits_opt 2
#python test_faiss_gpu.py  --checkpoint_path checkpoints/ckpt_quant/checkpoint-60 --output ckpt --quant_bits_opt 8
python test_faiss_gpu.py --only_recon --checkpoint_path checkpoints/qwen_ckpt_quant_pt_1000/checkpoint-500-ori --output ckpt --quant_bits_opt 8 --quant_stage 1
#python test_faiss.py --ref_checkpoint_path checkpoints/ckpt_quant_pt_1000/checkpoint-250 --checkpoint_path checkpoints/ckpt_quant_pt_1000/checkpoint-500 --output ckpt 
#python test_faiss.py --only_recon --ref_checkpoint_path checkpoints/ckpt_quant_pt_1000/checkpoint-250 --checkpoint_path checkpoints/ckpt_quant_pt_1000/checkpoint-500 --output ckpt 
#python test_faiss_gpu.py --checkpoint_path checkpoints/ckpt_quant_pt_1000/checkpoint-500 --output ckpt --quant_bits_opt 8 --quant_stage 2
