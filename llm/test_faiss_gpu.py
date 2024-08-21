import os
import copy
import glob
import time
import argparse
import numpy as np
import json
import paddle
import faiss

from safetensors import safe_open
from safetensors.numpy import save_file

paddle.set_device('cpu')

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录结束时间
        print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds to execute.")
        return result
    return wrapper

#@timeit
#def prune(tensor, mag_thres = 0.5):
#    mag_thres = min(2.5, mag_thres)
#    #mag_thres = paddle.median(paddle.abs(tensor)) * mag_thres
#    mag_thres = np.median(np.abs(tensor)) * mag_thres
#    tensor = np.where(np.abs(tensor) > mag_thres, tensor, 0)
#    return tensor, (tensor == 0).sum()

@timeit
def prune(tensor, mag_thres = 0.5):
    mag_thres = min(2.5, mag_thres)
    #tensor = tensor.numpy()
    mag_thres = paddle.median(paddle.abs(tensor)) * mag_thres
    #mag_thres = np.median(np.abs(tensor)) * mag_thres
    tensor = paddle.where(tensor.abs() > mag_thres, tensor, paddle.zeros_like(tensor))
    #tensor = np.where(np.abs(tensor) > mag_thres, tensor, 0)
    #tensor = paddle.Tensor(tensor, zero_copy=True)
    return tensor, (tensor == 0).sum()


#def prune_optimizer(exp_tensor, exp_sq_tensor, tensor, j = 1):
#    res_exp = np.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_tensor, np.zeros_like(exp_tensor))
#    res_exp = np.where(tensor == 0, np.zeros_like(res_exp), res_exp)
#    res_exp_sq = np.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_sq_tensor, np.zeros_like(tensor))
#    res_exp_sq = np.where(tensor == 0, np.zeros_like(res_exp_sq), res_exp_sq)
#    return res_exp, res_exp_sq, (res_exp == 0).sum()

@timeit
def prune_optimizer(exp_tensor, exp_sq_tensor, tensor, j = 1):
    res_exp = paddle.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_tensor, paddle.zeros_like(exp_tensor))
    res_exp = paddle.where(tensor == 0, paddle.zeros_like(res_exp), res_exp)
    res_exp_sq = paddle.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_sq_tensor, paddle.zeros_like(tensor))
    res_exp_sq = paddle.where(tensor == 0, paddle.zeros_like(res_exp_sq), res_exp_sq)
    return res_exp, res_exp_sq, (res_exp == 0).sum()

#@timeit
#def prune_optimizer(exp_tensor, exp_sq_tensor, tensor, j = 1):
#    res_exp = np.where((np.abs(exp_tensor)) > j * (np.abs(exp_tensor)).mean(), exp_tensor, 0)
#    res_exp = np.where(tensor == 0, 0, res_exp)
#    res_exp_sq = np.where((np.abs(exp_tensor)) > j * (np.abs(exp_tensor)).mean(), exp_sq_tensor, 0)
#    res_exp_sq = np.where(tensor == 0, 0, res_exp_sq)
#    return res_exp, res_exp_sq, (res_exp == 0).sum()

#@timeit
#def quant(tensor, bit_num = 4):
#    assert 8 % bit_num == 0 and bit_num <= 8
#    all_labels = []
#    all_codebook = []
#    knn_tensor = copy.deepcopy(tensor)
#    knn_tensor_shape = knn_tensor.shape
#    knn_tensor_flt = knn_tensor.flatten()
#    assert knn_tensor_flt.shape[0] % (8 // bit_num) == 0
#    knn_tensor_nonzero = knn_tensor_flt[knn_tensor_flt.nonzero()]
#    #print(knn_tensor_nonzero.shape[0])
#    if knn_tensor_nonzero.shape[0] <= 2 ** bit_num - 1:
#        return tensor, None
#    kmeans = faiss.Kmeans(1, 2 ** bit_num - 1, gpu=False)
#    kmeans.train(knn_tensor_nonzero.reshape([-1, 1]))
#    codebook = kmeans.centroids
#    labels = kmeans.index.search(knn_tensor_nonzero.reshape([-1, 1]), 1)[1]
#    knn_tensor_flt[knn_tensor_flt.nonzero()[0].squeeze()] = (labels + 1).squeeze()
#    #knn_tensor_flt[knn_tensor_flt != 0] = paddle.to_tensor(labels + 1).squeeze()
#    knn_tensor_flt2 = knn_tensor_flt.reshape([-1, 8 // bit_num])
#    knn_tensor_slim = np.zeros_like(knn_tensor_flt2[:, 0])
#    for i in range(8 // bit_num):
#        knn_tensor_slim += knn_tensor_flt2[:, i] * ((2 ** bit_num) ** (8 // bit_num - i - 1))
#    knn_tensor_flt = knn_tensor_slim.reshape([-1, knn_tensor_shape[-1] // (8 // bit_num)])
#    return knn_tensor_flt, codebook

#def quant(tensor, bit_num = 4):
#    assert 8 % bit_num == 0 and bit_num <= 8
#    all_labels = []
#    all_codebook = []
#    knn_tensor = copy.deepcopy(tensor)
#    knn_tensor_shape = knn_tensor.shape
#    knn_tensor_flt = knn_tensor.flatten()
#    assert knn_tensor_flt.shape[0] % (8 // bit_num) == 0
#    knn_tensor_nonzero = knn_tensor_flt[knn_tensor_flt.nonzero()]
#    #print(knn_tensor_nonzero.shape[0])
#    if knn_tensor_nonzero.shape[0] <= 2 ** bit_num - 1:
#        return tensor, None
#    kmeans = faiss.Kmeans(1, 2 ** bit_num - 1, gpu=False)
#    kmeans.train(knn_tensor_nonzero.reshape([-1, 1]).numpy())
#    codebook = kmeans.centroids
#    labels = kmeans.index.search(knn_tensor_nonzero.reshape([-1, 1]).numpy(), 1)[1]
#    knn_tensor_flt[knn_tensor_flt.nonzero().squeeze()] = paddle.to_tensor(labels + 1).squeeze()
#    #knn_tensor_flt[knn_tensor_flt != 0] = paddle.to_tensor(labels + 1).squeeze()
#    knn_tensor_flt2 = knn_tensor_flt.reshape([-1, 8 // bit_num])
#    knn_tensor_slim = paddle.zeros_like(knn_tensor_flt2[:, 0])
#    for i in range(8 // bit_num):
#        knn_tensor_slim += knn_tensor_flt2[:, i] * ((2 ** bit_num) ** (8 // bit_num - i - 1))
#    knn_tensor_flt = knn_tensor_slim.reshape([-1, knn_tensor_shape[-1] // (8 // bit_num)])
#    return knn_tensor_flt, codebook

def cal_abs_max_channel(inputs, quant_axis=1):
    reduce_axis = tuple(
        [i for i in range(len(inputs.shape)) if i != quant_axis])
    abs_max_values = np.max(np.abs(inputs), axis=reduce_axis)
    abs_max_values = np.where(
        abs_max_values == np.array(0, dtype=inputs.dtype),
        np.array(1e-8, dtype=inputs.dtype), abs_max_values)
    return abs_max_values

@timeit
def cal_radio(m, v, eps=1e-8):
    return (m/(np.sqrt(v) + eps)).astype(np.float16)

@timeit
def qdq_weight(x, quant_bit=8, quant_axis=-1, scales=None, dequant=False):
    if scales is None:
        scales = cal_abs_max_channel(x)
    bnt = (1 << (quant_bit - 1)) - 1
    if not dequant:
        # quant
        quant_x = np.clip(np.round(x / scales * bnt), -bnt - 1, bnt)
        return quant_x.astype(np.int8), scales
    else:
        quant_x = x
        # dequant
        qdq_x = quant_x / bnt * scales
        # fp32 , int8, int, fp32 or fp64
        print(quant_x.dtype, scales.dtype, bnt, qdq_x.dtype)
        #return qdq_x, scales
        return qdq_x.astype(np.float32), scales

#def cal_abs_max_channel(inputs, quant_axis=1):
#    reduce_axis = tuple(
#        [i for i in range(len(inputs.shape)) if i != quant_axis])
#    abs_max_values = paddle.max(paddle.abs(inputs), axis=reduce_axis)
#    abs_max_values = paddle.where(
#        abs_max_values == paddle.to_tensor(0, dtype=inputs.dtype),
#        paddle.to_tensor(1e-8, dtype=inputs.dtype), abs_max_values)
#    return abs_max_values
#
#@timeit
#def qdq_weight(x, quant_bit=8, quant_axis=-1, scales=None, dequant=False):
#    if scales is None:
#        scales = cal_abs_max_channel(x)
#    bnt = (1 << (quant_bit - 1)) - 1
#    if not dequant:
#        # quant
#        quant_x = paddle.clip(paddle.round(x / scales * bnt), -bnt - 1, bnt)
#        return quant_x.astype(paddle.int8), scales
#    else:
#        quant_x = x
#        # dequant
#        qdq_x = quant_x / bnt * scales
#        return qdq_x, scales

#@timeit
#def unquantize(codebook, indexes, bit_num):
#    recover_indexes = np.zeros([indexes.size, 8 // bit_num])
#    for i in range(8 // bit_num):
#        recover_indexes[:, i] = indexes.flatten() // ((2 ** bit_num) ** (8 // bit_num - i - 1))
#        indexes = indexes - (indexes // ((2 ** bit_num) ** (8 // bit_num - i - 1))) * ((2 ** bit_num) ** (8 // bit_num - i - 1))
#    tmp = np.array([0], dtype='float32')
#    #print([tmp, np.to_tensor(codebook.squeeze())])
#    recover_tensor = np.concatenate([tmp, np.array(codebook.squeeze())], axis=0)[recover_indexes.flatten().squeeze().astype(np.int64)]
#    return recover_tensor

#@timeit
#def unquantize(codebook, indexes, bit_num):
#    recover_indexes = paddle.zeros([indexes.numel(), 8 // bit_num])
#    for i in range(8 // bit_num):
#        recover_indexes[:, i] = indexes.flatten() // ((2 ** bit_num) ** (8 // bit_num - i - 1))
#        indexes = indexes - (indexes // ((2 ** bit_num) ** (8 // bit_num - i - 1))) * ((2 ** bit_num) ** (8 // bit_num - i - 1))
#    tmp = paddle.to_tensor([0], dtype='float32')
#    #print([tmp, paddle.to_tensor(codebook.squeeze())])
#    recover_tensor = paddle.concat([tmp, paddle.to_tensor(codebook.squeeze())], axis=0)[recover_indexes.flatten().squeeze().to(paddle.int64)]
#    return recover_tensor

def get_meta(checkpoint, i, args):
    optimizer_dict = {}
    for k in checkpoint.keys():
        k_m = k + '/moment1_0'
        k_v = k + '/moment2_0'
        optimizer_dict[k_v] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"
        optimizer_dict[k_m] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"

    return optimizer_dict

def recon(checkpoint, args, i):
    recon_dict = {}
    optimizer_dict = {}
    #data_pt = paddle.load(os.path.join(args.ref_checkpoint_path, "optimizer.pt"))
    data_pt = {}
    eps = 1e-8
    st = time.time()
    unquant_m, unquant_v, unquant_w, cast_w = 0, 0, 0, 0
    for k in checkpoint.keys():
        in_st = time.time()
        ckpt = checkpoint[k]
        #optim_k = optimizer_name.index(k)
        #ref_shape = ref_weights.get_tensor(k).shape
        #if "weights_c" in ckpt.keys():
        #    recover_weights = unquantize(ckpt["weights_c"].astype("float32"), ckpt["weights_i"].astype("int32"), args.quant_bits).reshape(ref_shape)
        #else:
        #    recover_weights = ckpt["weights_i"].reshape(ref_shape)
        unquant_w += time.time() - in_st
        in_st = time.time()
        #print(ckpt["opt_v_c"].dtype, ckpt["opt_v_i"].dtype)
        if "opt_m_c" in ckpt.keys():
            recover_opt_m, _ = qdq_weight(ckpt["opt_m_c"], scales=ckpt["opt_m_i"], quant_bit=args.quant_bits_opt, dequant=True)

        unquant_m += time.time() - in_st
        in_st = time.time()

        if args.quant_stage == 1:
            #recover_opt_v, _ = qdq_weight(ckpt["opt_v_c"], scales=ckpt["opt_v_i"], quant_bit=args.quant_bits_opt, dequant=True)
            recover_opt_v = np.square((recover_opt_m / (ckpt["opt_v_i"].astype(np.float32) + eps)) - eps)
        elif args.quant_stage == 2:
            recover_opt_m, recover_opt_v = np.split(recover_opt_m, 2, axis=0)
        elif args.quant_stage == 3:
            if ckpt["opt_v_i"].dtype == np.float16:
                recover_opt_v = np.square(ckpt["opt_v_i"].astype(np.float32))
            else:
                recover_opt_v, _ = qdq_weight(ckpt["opt_v_c"], scales=ckpt["opt_v_i"], quant_bit=args.quant_bits_opt, dequant=True)
                recover_opt_v = np.square(recover_opt_v)

        unquant_v += time.time() - in_st
        in_st = time.time()
        print(k, recover_opt_m.shape, recover_opt_m.dtype)
        #recon_dict[k] = (recover_weights + paddle.Tensor(ref_weights.get_tensor(k), zero_copy=True)).astype("bfloat16").numpy()
        #optim_k = optimizer_name.index(k)
        k_m = k + '/moment1_0'
        k_v = k + '/moment2_0'
        k_m_acc = k + '/beta1_pow_acc_0'
        k_v_acc = k + '/beta2_pow_acc_0'
        data_pt[k_v] = recover_opt_v
        data_pt[k_m] = recover_opt_m
        data_pt[k_v_acc] = ckpt["opt_v_acc"]
        data_pt[k_m_acc] = ckpt["opt_m_acc"]
        optimizer_dict[k_v] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"
        optimizer_dict[k_m] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"
        optimizer_dict[k_v_acc] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"
        optimizer_dict[k_m_acc] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"
        cast_w += time.time() - in_st
        in_st = time.time()
    all_time = unquant_w + unquant_v + unquant_m + cast_w
    print(f"unquant w: {unquant_w/all_time}, unquant v: {unquant_v/all_time}, unquant m: {unquant_m/all_time}, cast w: {cast_w/all_time}")
    #save_file(recon_dict, os.path.join(args.output, f"model-0000{i+1}-of-00008.safetensors"), metadata = {"format": "np"})
    save_file(data_pt, os.path.join(args.output, f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"), metadata = {"format": "np"})
    ed = time.time()
    print("ckpt save time: ",ed - st)
    return optimizer_dict

#def recon_1(checkpoint, args, i):
#    recon_dict = {}
#    optimizer_dict = {}
#    #data_pt = paddle.load(os.path.join(args.ref_checkpoint_path, "optimizer.pt"))
#    data_pt = {}
#    eps = 1e-8
#    st = time.time()
#    unquant_m, unquant_v, unquant_w, cast_w = 0, 0, 0, 0
#    for k in checkpoint.keys():
#        if not k.endswith('moment1_0') or not k.endswith('moment2_0'):
#            data_pt[k] = checkpoint.get_tensor(k)
#            continue
#        in_st = time.time()
#        ckpt = checkpoint.get_tensor(k)
#        cckpt = checkpoint.get_tensor(k + '_codebook')
#        #optim_k = optimizer_name.index(k)
#        #ref_shape = ref_weights.get_tensor(k).shape
#        #if "weights_c" in ckpt.keys():
#        #    recover_weights = unquantize(ckpt["weights_c"].astype("float32"), ckpt["weights_i"].astype("int32"), args.quant_bits).reshape(ref_shape)
#        #else:
#        #    recover_weights = ckpt["weights_i"].reshape(ref_shape)
#        unquant_w += time.time() - in_st
#        in_st = time.time()
#        #print(ckpt["opt_v_c"].dtype, ckpt["opt_v_i"].dtype)
#        recover_opt_m, _ = qdq_weight(ckpt, scales=cckpt, quant_bit=args.quant_bits_opt, dequant=True)
#
#        unquant_m += time.time() - in_st
#        in_st = time.time()
#
#        if args.quant_stage == 1:
#            #recover_opt_v, _ = qdq_weight(ckpt["opt_v_c"], scales=ckpt["opt_v_i"], quant_bit=args.quant_bits_opt, dequant=True)
#            recover_opt_v = np.square((recover_opt_m / (ckpt["opt_v_i"].astype(np.float32) + eps)) - eps)
#        elif args.quant_stage == 2:
#            recover_opt_m, recover_opt_v = np.split(recover_opt_m, 2, axis=0)
#        elif args.quant_stage == 3:
#            if ckpt.dtype == np.float16:
#                #recover_opt_v = np.square(ckpt.astype(np.float32))
#                recover_opt_v = ckpt.astype(np.float32)
#            else:
#                recover_opt_v, _ = qdq_weight(ckpt, scales=cckpt, quant_bit=args.quant_bits_opt, dequant=True)
#                recover_opt_v = np.square(recover_opt_v)
#
#        unquant_v += time.time() - in_st
#        in_st = time.time()
#        print(k, recover_opt_m.shape, recover_opt_m.dtype)
#        #recon_dict[k] = (recover_weights + paddle.Tensor(ref_weights.get_tensor(k), zero_copy=True)).astype("bfloat16").numpy()
#        #optim_k = optimizer_name.index(k)
#        k_m = k + '/moment1_0'
#        k_v = k + '/moment2_0'
#        k_m_acc = k + '/beta1_pow_acc_0'
#        k_v_acc = k + '/beta2_pow_acc_0'
#        data_pt[k_v] = recover_opt_v
#        data_pt[k_m] = recover_opt_m
#        optimizer_dict[k_v] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"
#        optimizer_dict[k_m] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"
#        optimizer_dict[k_v_acc] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"
#        optimizer_dict[k_m_acc] = f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"
#        cast_w += time.time() - in_st
#        in_st = time.time()
#    all_time = unquant_w + unquant_v + unquant_m + cast_w
#    print(f"unquant w: {unquant_w/all_time}, unquant v: {unquant_v/all_time}, unquant m: {unquant_m/all_time}, cast w: {cast_w/all_time}")
#    #save_file(recon_dict, os.path.join(args.output, f"model-0000{i+1}-of-00008.safetensors"), metadata = {"format": "np"})
#    save_file(data_pt, os.path.join(args.output, f"optimizer-0000{i+1}-of-0000{args.shard_num}.safetensors"), metadata = {"format": "np"})
#    ed = time.time()
#    print("ckpt save time: ",ed - st)
#    return optimizer_dict
class opt_handler:
    def __init__(self, index_file, ckpt_path):
        self.info = {}
        self.ckpt_path = ckpt_path
        self.file_io_map = {}
        with open(index_file, 'r') as f:
            self.info = json.load(f)
    
    def get_fd(self, key):
        file_name = self.info["weight_map"][key]
        if file_name not in self.file_io_map.keys():
            self.file_io_map[file_name] = safe_open(os.path.join(self.ckpt_path, file_name), framework="np", device="cpu")
        return self.file_io_map[file_name]

    def get_tensors(self, key):
        return self.get_fd(key).get_tensor(key)


def main(args):
    remove_counter_weights = 0
    remove_counter_optimizer = 0
    element_counter = 0
    #ref_handler = opt_handler(os.path.join(args.checkpoint_path, "optimizer.safetensors.index.json"), args.checkpoint_path)
    handler = opt_handler(os.path.join(args.checkpoint_path, "optimizer.safetensors.index.json"), args.checkpoint_path)
    #paddle.cuda.synchronize()
    st = time.time()
    eps = 1e-8
    #with safe_open(os.path.join(args.ref_checkpoint_path, "model-00001-of-00008.safetensors.safetensors"),\
    meta = {}
    quant_num, skip_num = 0, 0
    for i in range(args.start_idx, args.end_idx):
        print(f"saving model-0000{i+1}-of-0000{args.shard_num}.safetensors!")
        w_cast, opt_cast, prune_w, prune_opt, quant_w, quant_opt, save_cast = 0, 0, 0, 0, 0, 0, 0
        if args.only_recon:
            #saved_checkpoint = paddle.load(os.path.join(args.output, f"compressed_{i+1}.pt"))
            saved_checkpoint = safe_open(os.path.join(args.output, f"optimizer-0000{i+1}-of-00008.safetensors"), framework='np')[:]
            meta_data = recon(saved_checkpoint, args, i)
            #meta_data = get_meta(saved_checkpoint, i)
            meta.update(meta_data)
        else:
            with safe_open(os.path.join(args.checkpoint_path, f"model-0000{i+1}-of-0000{args.shard_num}.safetensors"),\
                                   framework="np", device="cpu") as weights:
                saved_checkpoint = {}
                for ind, k in enumerate(weights.keys()):
                    in_st = time.time()
                    #print(k)
                    #w = paddle.Tensor(weights.get_tensor(k), zero_copy=True)
                    #ref = paddle.Tensor(ref_weights.get_tensor(k), zero_copy=True)
                    #residual_tensor = (w - ref)
                    #residual_tensor = (w - ref).astype("float32").numpy()
                    w_cast += time.time() - in_st
                    in_st = time.time()
                    k_m = k + '/moment1_0'
                    k_v = k + '/moment2_0'
                    k_m_acc = k + '/beta1_pow_acc_0'
                    k_v_acc = k + '/beta2_pow_acc_0'
                    opt_m = handler.get_tensors(k_m)
                    opt_v = handler.get_tensors(k_v)
                    opt_m_acc = handler.get_tensors(k_m_acc)
                    opt_v_acc = handler.get_tensors(k_v_acc)
                    #opt_m = paddle.Tensor(np_m, zero_copy=True)
                    #opt_v = paddle.Tensor(np_v, zero_copy=True)
                    #opt_m = paddle.Tensor(np_m, zero_copy=True).astype("float32").numpy()
                    #opt_v = paddle.Tensor(np_v, zero_copy=True).astype("float32").numpy()
                    opt_cast += time.time() - in_st
                    in_st = time.time()
                    #residual_tensor, remove = prune(residual_tensor, args.prune_alpha / np.sqrt(opt_m.mean()))
                    #residual_tensor, remove = prune(residual_tensor, args.prune_alpha / opt_m.mean().sqrt())
                    prune_w += time.time() - in_st
                    in_st = time.time()
                    #pruned_opt_v, pruned_opt_m, remove_opt = prune_optimizer(opt_v, opt_m, residual_tensor, args.prune_beta)
                    pruned_opt_v, pruned_opt_m = opt_v, opt_m
                    prune_opt += time.time() - in_st
                    in_st = time.time()
                    #residual_tensor_index, residual_tensor_codebook = qdq_weight(residual_tensor, quant_bit=args.quant_bits)
                    quant_w += time.time() - in_st
                    in_st = time.time()
                    #remove_counter_weights += remove
                    #remove_counter_optimizer += remove_opt
                    #element_counter += residual_tensor.numel()
                    opt_m_index, opt_m_codebook, opt_v_index, opt_v_codebook = None, None, None, None
                    if args.quant_stage == 1:
                        #opt_v_index, opt_v_codebook = qdq_weight(pruned_opt_v, quant_bit=args.quant_bits_opt)
                        opt_v_index = cal_radio(pruned_opt_m, pruned_opt_v)
                        opt_m_index, opt_m_codebook = qdq_weight(pruned_opt_m, quant_bit=args.quant_bits_opt)
                    elif args.quant_stage == 2:
                        concat_mv = np.concatenate((pruned_opt_m, np.sqrt(pruned_opt_v)), axis=0)
                        opt_m_index, opt_m_codebook = qdq_weight(concat_mv, quant_bit=args.quant_bits_opt)
                        #opt_v_index, opt_v_codebook = qdq_weight(1/(np.sqrt(pruned_opt_v) + eps), quant_bit=args.quant_bits_opt)
                        #qdq_weight(opt_m_codebook, scales=opt_m_index, quant_bit=args.quant_bits_opt, dequant=True)
                    elif args.quant_stage == 3:
                        opt_m_index, opt_m_codebook = qdq_weight(pruned_opt_m, quant_bit=args.quant_bits_opt)

                        sqrt_v = np.sqrt(pruned_opt_v)
                        opt_v_index, opt_v_codebook = qdq_weight(sqrt_v, quant_bit=args.quant_bits_opt)
                        if args.fusion_quant:
                            # peek dequant elems
                            peek_dequant, _ = qdq_weight(opt_v_codebook, scales=opt_v_index, quant_bit=args.quant_bits_opt, dequant=True)
                            # non zero mask
                            nonzero_mask = sqrt_v != 0.0
                            # outlier flag
                            has_outlier = not np.all(peek_dequant[nonzero_mask] != 0.0)
                            if has_outlier:
                                #print(f"{k_v} has outliers cnt {peek_dequant[nonzero_mask][peek_dequant[nonzero_mask] == 0].shape}, skip.")
                                skip_num += 1
                                #import pdb; pdb.set_trace()
                                opt_v_index = sqrt_v.astype(np.float16)
                            else:
                                print(f"{k_v} dont has outliers.")
                                quant_num += 1
                    
                    quant_opt += time.time() - in_st
                    in_st = time.time()
                    saved_checkpoint[k] = {}
                    #saved_checkpoint[k]['weights_i'] = residual_tensor_index
                    #if residual_tensor_codebook is not None:
                    #    saved_checkpoint[k]['weights_i'] = residual_tensor_index
                    #    saved_checkpoint[k]["weights_c"] = residual_tensor_codebook
                    saved_checkpoint[k]["opt_v_i"] = opt_v_index
                    saved_checkpoint[k]["opt_v_c"] = opt_v_codebook
                    saved_checkpoint[k]["opt_v_acc"] = opt_v_acc
                    saved_checkpoint[k]["opt_m_i"] = opt_m_index
                    saved_checkpoint[k]["opt_m_c"] = opt_m_codebook
                    saved_checkpoint[k]["opt_m_acc"] = opt_m_acc
                    #saved_checkpoint[k]["opt_v_i"] = opt_v_index
                    #if opt_v_codebook is not None:
                    #    saved_checkpoint[k]["opt_v_i"] = opt_v_index
                    #    saved_checkpoint[k]["opt_v_c"] = opt_v_codebook
                    #saved_checkpoint[k]["opt_m_i"] = opt_m_index
                    #if opt_m_codebook is not None:
                    #    saved_checkpoint[k]["opt_m_i"] = opt_m_index
                    #    saved_checkpoint[k]["opt_m_c"] = opt_m_codebook
                    #save_cast += time.time() - in_st
                    in_st = time.time()
                    #all_time = w_cast + opt_cast + prune_w + prune_opt + quant_w + quant_opt + save_cast
                    #print(f'weight cast: {w_cast/all_time}, opt cast: {opt_cast/all_time}, prune w: {prune_w/all_time},prune opt: {prune_opt/all_time}, quant w: {quant_w/all_time}, quant opt: {quant_opt/all_time}, save cast: {save_cast/all_time},')

                ed = time.time()
                print("compress using time: {}".format(ed - st))
                #import pdb; pdb.set_trace()
                #print("weights removed ratio: {}/{}({})".format(remove_counter_weights.item(), element_counter.item(), remove_counter_weights.item() / element_counter.item()))
                #print("optimizer removed ratio: {}/{}({})".format(remove_counter_optimizer.item(), element_counter.item(), remove_counter_optimizer.item() / element_counter.item()))
                #print("weights removed ratio: {}/{}({})".format(remove_counter_weights, element_counter, remove_counter_weights / element_counter))
                #print("optimizer removed ratio: {}/{}({})".format(remove_counter_optimizer, element_counter, remove_counter_optimizer / element_counter))
                if not os.path.exists(args.output):
                    os.makedirs(args.output, exist_ok=True)
                #paddle.save(saved_checkpoint, os.path.join(args.output, f"compressed_{i+1}.pt"))
                paddle.save(saved_checkpoint, os.path.join(args.output, f"compressed_{i+1}.pt"))
                del saved_checkpoint
                save_time = time.time() - ed
            all_time = w_cast + opt_cast + prune_w + prune_opt + quant_w + quant_opt + save_cast + save_time
            #import pdb; pdb.set_trace()
            print(f'weight cast: {w_cast/all_time}, opt cast: {opt_cast/all_time}, prune w: {prune_w/all_time},prune opt: {prune_opt/all_time}, quant w: {quant_w/all_time}, quant opt: {quant_opt/all_time}, save cast: {save_cast/all_time}, save time: {save_time/all_time}')
    st = time.time()
    if args.only_recon:
        info_ = copy.deepcopy(handler.info)
        info_["weight_map"]=meta
        #import pdb; pdb.set_trace()
        with open(os.path.join(args.output, "optimizer.safetensors.index.json"), 'w') as meta_d:
            json.dump(
                info_,
                meta_d,
                indent=4,
            )
    if args.recon:
        for i in range(8):
            saved_ckpt = paddle.load(os.path.join(args.output, f"compressed_{i+1}.pt"))
            meta_data = recon(saved_ckpt, args, i)
    #paddle.cuda.synchronize()
    ed = time.time()
    print(f"quant key {quant_num}, skip key {skip_num}.")
    print("recon using time: {}".format(ed - st))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test a SAM')
    parser.add_argument('checkpoint_path', type=str, default="checkpoints/llama_sft_ckpts/checkpoint-2")
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--prune_alpha', type=float, default=5e-5)
    parser.add_argument('--prune_beta', type=float, default=2.0)
    parser.add_argument('--quant_bits', type=int, default=8)
    parser.add_argument('--quant_stage', type=int, default=1)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=8)
    parser.add_argument('--shard_num', type=int, default=4)
    parser.add_argument('--quant_bits_opt', type=int, default=8)
    parser.add_argument('--fusion_quant', action='store_true')
    parser.add_argument('--recon', action='store_true')
    parser.add_argument('--only_recon', action='store_true')
    parser.add_argument('--output', type=str, default="./")
    args, _ = parser.parse_known_args()
    main(args)

#aaa = paddle.randn([100, 10])
#out1, out2 = quant(aaa, 4)
#out = unquantize(out2, out1, 4)
#out = out.reshape(aaa.shape)
#import pdb; pdb.set_trace()
#
