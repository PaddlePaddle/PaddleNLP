# Data directories
train_data_path = "./finished_files/chunked/train_*.json"
eval_data_path = "./finished_files/val.json"
decode_data_path = "./finished_files/test.json"
vocab_path = "./finished_files/vocab"
log_root = "./log"

# Hyperparameters
hidden_dim = 256
emb_dim = 128
batch_size = 8
max_enc_steps = 400
max_dec_steps = 100
beam_size = 4
min_dec_steps = 35
vocab_size = 50000

lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

pointer_gen = True
is_coverage = True
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 100000

lr_coverage = 0.15
