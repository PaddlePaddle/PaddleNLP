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

@timeit
def prune(tensor, mag_thres = 0.5):
    mag_thres = min(2.5, mag_thres)
    #mag_thres = paddle.median(paddle.abs(tensor)) * mag_thres
    mag_thres = np.median(np.abs(tensor)) * mag_thres
    tensor = np.where(np.abs(tensor) > mag_thres, tensor, 0)
    return tensor, (tensor == 0).sum()

#@timeit
#def prune(tensor, mag_thres = 0.5):
#    mag_thres = min(2.5, mag_thres)
#    #tensor = tensor.numpy()
#    mag_thres = paddle.median(paddle.abs(tensor)) * mag_thres
#    #mag_thres = np.median(np.abs(tensor)) * mag_thres
#    tensor = paddle.where(tensor.abs() > mag_thres, tensor, paddle.zeros_like(tensor))
#    #tensor = np.where(np.abs(tensor) > mag_thres, tensor, 0)
#    #tensor = paddle.Tensor(tensor, zero_copy=True)
#    return tensor, (tensor == 0).sum()


#def prune_optimizer(exp_tensor, exp_sq_tensor, tensor, j = 1):
#    res_exp = np.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_tensor, np.zeros_like(exp_tensor))
#    res_exp = np.where(tensor == 0, np.zeros_like(res_exp), res_exp)
#    res_exp_sq = np.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_sq_tensor, np.zeros_like(tensor))
#    res_exp_sq = np.where(tensor == 0, np.zeros_like(res_exp_sq), res_exp_sq)
#    return res_exp, res_exp_sq, (res_exp == 0).sum()

#def prune_optimizer(exp_tensor, exp_sq_tensor, tensor, j = 1):
#    res_exp = paddle.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_tensor, paddle.zeros_like(exp_tensor))
#    res_exp = paddle.where(tensor == 0, paddle.zeros_like(res_exp), res_exp)
#    res_exp_sq = paddle.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_sq_tensor, paddle.zeros_like(tensor))
#    res_exp_sq = paddle.where(tensor == 0, paddle.zeros_like(res_exp_sq), res_exp_sq)
#    return res_exp, res_exp_sq, (res_exp == 0).sum()

@timeit
def prune_optimizer(exp_tensor, exp_sq_tensor, tensor, j = 1):
    res_exp = np.where((np.abs(exp_tensor)) > j * (np.abs(exp_tensor)).mean(), exp_tensor, 0)
    res_exp = np.where(tensor == 0, 0, res_exp)
    res_exp_sq = np.where((np.abs(exp_tensor)) > j * (np.abs(exp_tensor)).mean(), exp_sq_tensor, 0)
    res_exp_sq = np.where(tensor == 0, 0, res_exp_sq)
    return res_exp, res_exp_sq, (res_exp == 0).sum()

@timeit
def quant(tensor, bit_num = 4):
    assert 8 % bit_num == 0 and bit_num <= 8
    all_labels = []
    all_codebook = []
    knn_tensor = copy.deepcopy(tensor)
    knn_tensor_shape = knn_tensor.shape
    knn_tensor_flt = knn_tensor.flatten()
    assert knn_tensor_flt.shape[0] % (8 // bit_num) == 0
    knn_tensor_nonzero = knn_tensor_flt[knn_tensor_flt.nonzero()]
    #print(knn_tensor_nonzero.shape[0])
    if knn_tensor_nonzero.shape[0] <= 2 ** bit_num - 1:
        return tensor, None
    kmeans = faiss.Kmeans(1, 2 ** bit_num - 1, gpu=False)
    kmeans.train(knn_tensor_nonzero.reshape([-1, 1]))
    codebook = kmeans.centroids
    labels = kmeans.index.search(knn_tensor_nonzero.reshape([-1, 1]), 1)[1]
    knn_tensor_flt[knn_tensor_flt.nonzero()[0].squeeze()] = (labels + 1).squeeze()
    #knn_tensor_flt[knn_tensor_flt != 0] = paddle.to_tensor(labels + 1).squeeze()
    knn_tensor_flt2 = knn_tensor_flt.reshape([-1, 8 // bit_num])
    knn_tensor_slim = np.zeros_like(knn_tensor_flt2[:, 0])
    for i in range(8 // bit_num):
        knn_tensor_slim += knn_tensor_flt2[:, i] * ((2 ** bit_num) ** (8 // bit_num - i - 1))
    knn_tensor_flt = knn_tensor_slim.reshape([-1, knn_tensor_shape[-1] // (8 // bit_num)])
    #import pdb; pdb.set_trace()
    return knn_tensor_flt, codebook

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


@timeit
def unquantize(codebook, indexes, bit_num):
    recover_indexes = np.zeros([indexes.size, 8 // bit_num])
    for i in range(8 // bit_num):
        recover_indexes[:, i] = indexes.flatten() // ((2 ** bit_num) ** (8 // bit_num - i - 1))
        indexes = indexes - (indexes // ((2 ** bit_num) ** (8 // bit_num - i - 1))) * ((2 ** bit_num) ** (8 // bit_num - i - 1))
    tmp = np.array([0], dtype='float32')
    #print([tmp, np.to_tensor(codebook.squeeze())])
    recover_tensor = np.concatenate([tmp, np.array(codebook.squeeze())], axis=0)[recover_indexes.flatten().squeeze().astype(np.int64)]
    return recover_tensor

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

def get_meta(checkpoint, i):
    optimizer_dict = {}
    for k in checkpoint.keys():
        k_m = k + '/moment1_0'
        k_v = k + '/moment2_0'
        optimizer_dict[k_v] = f"model-0000{i+1}-of-00008.safetensors"
        optimizer_dict[k_m] = f"model-0000{i+1}-of-00008.safetensors"

    return optimizer_dict

def recon(checkpoint, ref_weights, args, i):
    recon_dict = {}
    optimizer_dict = {}
    #data_pt = paddle.load(os.path.join(args.ref_checkpoint_path, "optimizer.pt"))
    data_pt = {}
    st = time.time()
    unquant_m, unquant_v, unquant_w, cast_w = 0, 0, 0, 0
    for k in checkpoint.keys():
        in_st = time.time()
        ckpt = checkpoint[k]
        #optim_k = optimizer_name.index(k)
        ref_shape = ref_weights.get_tensor(k).shape
        if "weights_c" in ckpt.keys():
            recover_weights = unquantize(ckpt["weights_c"].astype("float32"), ckpt["weights_i"].astype("int32"), args.quant_bits).reshape(ref_shape)
        else:
            recover_weights = ckpt["weights_i"].reshape(ref_shape)
        unquant_w += time.time() - in_st
        in_st = time.time()
        if "opt_v_c" in ckpt.keys():
            recover_opt_v = unquantize(ckpt["opt_v_c"].astype("float32"), ckpt["opt_v_i"].astype("int32"), args.quant_bits_opt).reshape(ref_shape)
        else:
            recover_opt_v = ckpt["opt_v_i"].reshape(ref_shape)
        unquant_v += time.time() - in_st
        in_st = time.time()
        if "opt_m_c" in ckpt.keys():
            recover_opt_m = unquantize(ckpt["opt_m_c"].astype("float32"), ckpt["opt_m_i"].astype("int32"), args.quant_bits_opt).reshape(ref_shape)
        else:
            recover_opt_m = ckpt["opt_m_i"].reshape(ref_shape)
        unquant_m += time.time() - in_st
        in_st = time.time()
        print(k, recover_weights.shape, ref_weights.get_tensor(k).shape, recover_weights.dtype)
        recon_dict[k] = (recover_weights + paddle.Tensor(ref_weights.get_tensor(k), zero_copy=True)).astype("bfloat16").numpy()
        #optim_k = optimizer_name.index(k)
        k_m = k + '/moment1_0'
        k_v = k + '/moment2_0'
        data_pt[k_v] = recover_opt_v
        data_pt[k_m] = recover_opt_m
        optimizer_dict[k_v] = f"model-0000{i+1}-of-00008.safetensors"
        optimizer_dict[k_m] = f"model-0000{i+1}-of-00008.safetensors"
        cast_w += time.time() - in_st
        in_st = time.time()
    all_time = unquant_w + unquant_v + unquant_m + cast_w
    print(f"unquant w: {unquant_w/all_time}, unquant v: {unquant_v/all_time}, unquant m: {unquant_m/all_time}, cast w: {cast_w/all_time}")
    save_file(recon_dict, os.path.join(args.output, f"model-0000{i+1}-of-00008.safetensors"), metadata = {"format": "np"})
    save_file(data_pt, os.path.join(args.output, f"optimizer-0000{i+1}-of-00008.safetensors"), metadata = {"format": "np"})
    ed = time.time()
    print("ckpt save time: ",ed - st)
    return optimizer_dict

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
    #with safe_open(os.path.join(args.ref_checkpoint_path, "model-00001-of-00008.safetensors.safetensors"),\
    meta = {}
    for i in range(8):
        print(f"saving model-0000{i+1}-of-00008.safetensors!")
        w_cast, opt_cast, prune_w, prune_opt, quant_w, quant_opt, save_cast = 0, 0, 0, 0, 0, 0, 0
        with safe_open(os.path.join(args.ref_checkpoint_path, f"model-0000{i+1}-of-00008.safetensors"),\
                                   framework="np", device="cpu") as ref_weights:
            if args.only_recon:
                saved_checkpoint = paddle.load(os.path.join(args.output, f"compressed_{i+1}.pt"))
                meta_data = recon(saved_checkpoint, ref_weights, args, i)
                #meta_data = get_meta(saved_checkpoint, i)
                meta.update(meta_data)
            else:
                with safe_open(os.path.join(args.checkpoint_path, f"model-0000{i+1}-of-00008.safetensors"),\
                                       framework="np", device="cpu") as weights:
                    saved_checkpoint = {}
                    for ind, k in enumerate(ref_weights.keys()):
                        in_st = time.time()
                        #print(k)
                        w = paddle.Tensor(weights.get_tensor(k), zero_copy=True)
                        ref = paddle.Tensor(ref_weights.get_tensor(k), zero_copy=True)
                        #residual_tensor = (w - ref).astype("float32")
                        residual_tensor = (w - ref).astype("float32").numpy()
                        w_cast += time.time() - in_st
                        in_st = time.time()
                        k_m = k + '/moment1_0'
                        k_v = k + '/moment2_0'
                        opt_m = handler.get_tensors(k_m)
                        opt_v = handler.get_tensors(k_v)
                        #opt_m = paddle.Tensor(np_m, zero_copy=True).astype("float32")
                        #opt_v = paddle.Tensor(np_v, zero_copy=True).astype("float32")
                        #opt_m = paddle.Tensor(np_m, zero_copy=True).astype("float32").numpy()
                        #opt_v = paddle.Tensor(np_v, zero_copy=True).astype("float32").numpy()
                        opt_cast += time.time() - in_st
                        in_st = time.time()
                        residual_tensor, remove = prune(residual_tensor, args.prune_alpha / np.sqrt(opt_m.mean()))
                        #residual_tensor, remove = prune(residual_tensor, args.prune_alpha / opt_m.mean().sqrt())
                        prune_w += time.time() - in_st
                        in_st = time.time()
                        pruned_opt_v, pruned_opt_m, remove_opt = prune_optimizer(opt_v, opt_m, residual_tensor, args.prune_beta)
                        prune_opt += time.time() - in_st
                        in_st = time.time()
                        residual_tensor_index, residual_tensor_codebook = quant(residual_tensor, args.quant_bits)
                        quant_w += time.time() - in_st
                        in_st = time.time()
                        remove_counter_weights += remove
                        remove_counter_optimizer += remove_opt
                        element_counter += residual_tensor.size
                        opt_v_index, opt_v_codebook = quant(pruned_opt_v, args.quant_bits_opt)
                        opt_m_index, opt_m_codebook = quant(pruned_opt_m, args.quant_bits_opt)
                        quant_opt += time.time() - in_st
                        in_st = time.time()
                        saved_checkpoint[k] = {}
                        saved_checkpoint[k]['weights_i'] = residual_tensor_index
                        if residual_tensor_codebook is not None:
                            saved_checkpoint[k]['weights_i'] = residual_tensor_index.astype(np.uint8)
                            saved_checkpoint[k]["weights_c"] = residual_tensor_codebook.astype(np.float16)
                        saved_checkpoint[k]["opt_v_i"] = opt_v_index
                        if opt_v_codebook is not None:
                            saved_checkpoint[k]["opt_v_i"] = opt_v_index.astype(np.uint8)
                            saved_checkpoint[k]["opt_v_c"] = opt_v_codebook.astype(np.float16)
                        saved_checkpoint[k]["opt_m_i"] = opt_m_index
                        if opt_m_codebook is not None:
                            saved_checkpoint[k]["opt_m_i"] = opt_m_index.astype(np.uint8)
                            saved_checkpoint[k]["opt_m_c"] = opt_m_codebook.astype(np.float16)
                        #saved_checkpoint[k]['weights_i'] = paddle.to_tensor(residual_tensor_index)
                        #if residual_tensor_codebook is not None:
                        #    saved_checkpoint[k]['weights_i'] = paddle.to_tensor(residual_tensor_index).to(paddle.uint8)
                        #    saved_checkpoint[k]["weights_c"] = paddle.to_tensor(residual_tensor_codebook).to(paddle.float16)
                        #saved_checkpoint[k]["opt_v_i"] = paddle.to_tensor(opt_v_index)
                        #if opt_v_codebook is not None:
                        #    saved_checkpoint[k]["opt_v_i"] = paddle.to_tensor(opt_v_index).to(paddle.uint8)
                        #    saved_checkpoint[k]["opt_v_c"] = paddle.to_tensor(opt_v_codebook).to(paddle.float16)
                        #saved_checkpoint[k]["opt_m_i"] = paddle.to_tensor(opt_m_index)
                        #if opt_m_codebook is not None:
                        #    saved_checkpoint[k]["opt_m_i"] = paddle.to_tensor(opt_m_index).to(paddle.uint8)
                        #    saved_checkpoint[k]["opt_m_c"] = paddle.to_tensor(opt_m_codebook).to(paddle.float16)
                        save_cast += time.time() - in_st
                        in_st = time.time()
                        #all_time = w_cast + opt_cast + prune_w + prune_opt + quant_w + quant_opt + save_cast
                        #print(f'weight cast: {w_cast/all_time}, opt cast: {opt_cast/all_time}, prune w: {prune_w/all_time},prune opt: {prune_opt/all_time}, quant w: {quant_w/all_time}, quant opt: {quant_opt/all_time}, save cast: {save_cast/all_time},')

                    ed = time.time()
                    print("compress using time: {}".format(ed - st))
                    #print("weights removed ratio: {}/{}({})".format(remove_counter_weights.item(), element_counter.item(), remove_counter_weights.item() / element_counter.item()))
                    #print("optimizer removed ratio: {}/{}({})".format(remove_counter_optimizer.item(), element_counter.item(), remove_counter_optimizer.item() / element_counter.item()))
                    print("weights removed ratio: {}/{}({})".format(remove_counter_weights, element_counter, remove_counter_weights / element_counter))
                    print("optimizer removed ratio: {}/{}({})".format(remove_counter_optimizer, element_counter, remove_counter_optimizer / element_counter))
                    if not os.path.exists(args.output):
                        os.makedirs(args.output, exist_ok=True)
                    #paddle.save(saved_checkpoint, os.path.join(args.output, f"compressed_{i+1}.pt"))
                    paddle.save(saved_checkpoint, os.path.join(args.output, f"compressed_{i+1}.pt"))
                    del saved_checkpoint
                all_time = w_cast + opt_cast + prune_w + prune_opt + quant_w + quant_opt + save_cast
                print(f'weight cast: {w_cast/all_time}, opt cast: {opt_cast/all_time}, prune w: {prune_w/all_time},prune opt: {prune_opt/all_time}, quant w: {quant_w/all_time}, quant opt: {quant_opt/all_time}, save cast: {save_cast/all_time},')
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
            with safe_open(os.path.join(args.ref_checkpoint_path, f"model-0000{i+1}-of-00008.safetensors"),\
                                       framework="np", device="cpu") as ref:
                saved_ckpt = paddle.load(os.path.join(args.output, f"compressed_{i+1}.pt"))
                meta_data = recon(saved_ckpt, ref, args, i, handler.info)
    #paddle.cuda.synchronize()
    ed = time.time()
    print("recon using time: {}".format(ed - st))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test a SAM')
    parser.add_argument('checkpoint_path', type=str, default="checkpoints/llama_sft_ckpts/checkpoint-2")
    parser.add_argument('ref_checkpoint_path', type=str, default="checkpoints/llama_sft_ckpts/checkpoint-1")
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--prune_alpha', type=float, default=5e-5)
    parser.add_argument('--prune_beta', type=float, default=2.0)
    parser.add_argument('--quant_bits', type=int, default=4)
    parser.add_argument('--quant_bits_opt', type=int, default=4)
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
