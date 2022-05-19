# !/bin/sh
for seq_len in 32 64 128 256 512; do
for batch_size in 1 2 4 8 16 32 64; do
mkdir -p seq_len_$seq_len/batch_size_$batch_size
for thread_num in 1 2 4 8 16 32 64; do
echo "Experiment setting: thread_num=$thread_num, batch_size=$batch_size, sequence_length=$seq_len"
export OMP_NUM_THREADS=$thread_num
export RAYON_RS_NUM_CPUS=$thread_num
python perf.py --batch_size $batch_size --max_seq_length $seq_len >seq_len_$seq_len/batch_size_$batch_size/parallel$thread_num.log 2>nohup.out
done 
done
done