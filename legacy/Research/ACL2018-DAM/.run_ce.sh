export CE_MODE_X=ce

export CUDA_VISIBLE_DEVICES=0

python -u main.py \
  --do_train True \
    --use_cuda \
      --data_path ./data/ubuntu/data_small.pkl \
        --save_path ./model_files/ubuntu \
          --use_pyreader \
            --vocab_size 434512 \
              --_EOS_ 28270 \
                --batch_size 32 | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u main.py \
  --do_train True \
    --use_cuda \
      --data_path ./data/ubuntu/data_small.pkl \
        --save_path ./model_files/ubuntu \
          --use_pyreader \
            --vocab_size 434512 \
              --_EOS_ 28270 \
                --batch_size 32 | python _ce.py
