import numpy as np
import paddle
import sys
import time
try:
    import deep_gemm
    import kitchen
    import kitchen.quantization_subchannel_block_hybrid
    from kitchen.quantization import QParams, ScalingType
except:
    pass

from paddle.base import core

from paddle.incubate.nn.functional import swiglu
import TokenDispatcherUtils as TDU

H1 = 7168
H2 = 2048
topk = 8


topk_ind = np.load("topk_indice.npy")
reci_x = paddle.randn( [ topk_ind.shape[0], H1], dtype="bfloat16")
reci_x_fp8 = reci_x.cast("float8_e4m3fn")
reci_x_scale = paddle.randn((reci_x.shape[0], int((H1 + 127) / 128)), dtype="float32")
print("reci_x scale shape: ", reci_x_scale.shape)
topk_ind_base = paddle.to_tensor(topk_ind, dtype="int32")
probs = paddle.ones( topk_ind_base.shape, dtype="bfloat16") # uses ones as topk_ind_base???
print( topk_ind.shape)
print( "recv x", reci_x.shape)

total_num = int((topk_ind != -1).astype("int64").sum())

e0_num = int( (topk_ind == 0).astype("int64").sum() )
e1_num = int((topk_ind == 1).astype("int64").sum())
e2_num = int( (topk_ind == 2).astype("int64").sum())
e3_num = int( (topk_ind == 3).astype("int64").sum())
token_per_expert = [  e0_num, e1_num, e2_num, e3_num ]
print("token per expert: ", token_per_expert)
max_tokens = max(token_per_expert)



def test_unzip_stable():
    # ---------------------------- Forward --------------------
    # ------------ unzip and preprocess --------------
    # ----------- FP8 -----------------
    print("############## FP8 ################")
    unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, unzipped_scales = TDU.tokens_unzip_stable(
        reci_x_fp8,
        reci_x_scale,
        topk_ind_base, 
        probs,
        topk=topk, num_experts=4, max_tokens_per_expert=max_tokens)
    
    print("zipped_expertwise_rowmap_fp8: ", zipped_expertwise_rowmap)
    np.savetxt("zipped_expertwise_rowmap_fp8.csv", zipped_expertwise_rowmap, delimiter=",", fmt='%d')
    np.savetxt("topk_ind_fp8.csv", topk_ind, delimiter=",", fmt='%d')
    np.savetxt("unzipped_scales_fp8.csv", unzipped_scales[:10, :], delimiter=",", fmt='%d')
    # ----------- BF16 -----------------
    print("############## BF16 ################")
    unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, _= TDU.tokens_unzip_stable(
        reci_x,
        None,
        topk_ind_base, 
        probs,
        topk=topk, num_experts=4, max_tokens_per_expert=max_tokens)
    
    print("zipped_expertwise_rowmap: ", zipped_expertwise_rowmap)
    np.savetxt("zipped_expertwise_rowmap.csv", zipped_expertwise_rowmap, delimiter=",", fmt='%d')
    np.savetxt("topk_ind.csv", topk_ind, delimiter=",", fmt='%d')

# core.nvprof_enable_record_event()

for i in range(1):
    core.nvprof_nvtx_push("step")
    if i == 1:
      core.nvprof_start()
    test_unzip_stable()
    core.nvprof_nvtx_pop()