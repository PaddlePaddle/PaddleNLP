import numpy as np
import paddle
import time
import FusedQuantOps as FQO


def gen_m_indices_cpu(tokens_per_expert):
    tokens = []
    for i in range(len(tokens_per_expert)):
        tokens.append(paddle.full([tokens_per_expert[i]], i, dtype="int32"))
    out = paddle.concat(tokens, axis=0)
    return out


def gen_m_indices_gpu_fast(tokens_per_expert):
    result = FQO.generate_expert_indices_fast(tokens_per_expert)
    return result


#验证正确性
def compare_expert_indices(cpu_result, gpu_result, tokens_per_expert):
    cpu_np = cpu_result.numpy()
    gpu_np = gpu_result.numpy()
    
    assert cpu_np.shape == gpu_np.shape, f"Shape mismatch: CPU {cpu_np.shape} vs GPU {gpu_np.shape}"
    
    if np.array_equal(cpu_np, gpu_np):
        print("Arrays Match")
        return True
    
    diff_count = np.sum(cpu_np != gpu_np)
    print(f"Arrays differ in {diff_count}/{len(cpu_np)} positions")
    
    all_counts_correct = True
    for expert_idx in range(len(tokens_per_expert)):
        cpu_count = np.sum(cpu_np == expert_idx)
        gpu_count = np.sum(gpu_np == expert_idx)
        expected = tokens_per_expert[expert_idx]
        
        if cpu_count != expected or gpu_count != expected:
            print(f"Expert {expert_idx}: CPU={cpu_count}, GPU={gpu_count}, Expected={expected}")
            all_counts_correct = False
    
    return all_counts_correct

#对比速度
def benchmark_implementations(tokens_per_expert, num_iterations=10):
    """Benchmark CPU and ultra-fast GPU implementations"""
    total_tokens = sum(tokens_per_expert)
    
    # 预热
    gen_m_indices_cpu(tokens_per_expert)
    
    cpu_times = []
    for _ in range(num_iterations):
        start = time.time()
        cpu_result = gen_m_indices_cpu(tokens_per_expert)
        end = time.time()
        cpu_times.append(end - start)
    
    cpu_time = np.mean(cpu_times)
    cpu_throughput = total_tokens / (cpu_time * 1000)
    
    #预热
    gen_m_indices_gpu_fast(tokens_per_expert)
    if paddle.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()
    
    gpu_fast_times = []
    for _ in range(num_iterations):
        start = time.time()
        gpu_fast_result = gen_m_indices_gpu_fast(tokens_per_expert)
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
        end = time.time()
        gpu_fast_times.append(end - start)
    
    gpu_fast_time = np.mean(gpu_fast_times)
    gpu_fast_throughput = total_tokens / (gpu_fast_time * 1000)
    
    fast_speedup = cpu_time / gpu_fast_time
    
    return {
        'cpu_time': cpu_time,
        'gpu_fast_time': gpu_fast_time,
        'cpu_throughput': cpu_throughput,
        'gpu_fast_throughput': gpu_fast_throughput,
        'fast_speedup': fast_speedup,
        'cpu_result': cpu_result,
        'gpu_fast_result': gpu_fast_result
    }


def verify_expert_indices_result():
    """Test expert indices generation with different configurations"""
    test_configs = [
        ([32768] * 4, "4 experts × 32K tokens"),
        ([32768] * 8, "8 experts × 32K tokens"),
        ([32768] * 16, "16 experts × 32K tokens"),
        ([32768] * 32, "32 experts × 32K tokens"),
    ]
    
    print("=" * 80)
    print("EXPERT INDICES GENERATION VERIFICATION")
    print("=" * 80)
    
    for tokens_per_expert, description in test_configs:
        total_tokens = sum(tokens_per_expert)
        print(f"\n{'#' * 20} Testing {description} {'#' * 20}")
        print(f"Configuration: {len(tokens_per_expert)} experts, {total_tokens:,} total tokens")
        
        try:
            results = benchmark_implementations(tokens_per_expert)
            
            print(f"\nPerformance Results:")
            print(f"  CPU time:           {results['cpu_time']*1000:.2f} ms")
            print(f"  GPU Fast time:      {results['gpu_fast_time']*1000:.2f} ms")
            print(f"  Speedup:            {results['fast_speedup']:.1f}x")
            print(f"  CPU throughput:     {results['cpu_throughput']:,.0f} tokens/ms")
            print(f"  GPU Fast throughput: {results['gpu_fast_throughput']:,.0f} tokens/ms")
            
            print(f"\nCorrectness Check:")
            is_correct = compare_expert_indices(
                results['cpu_result'], 
                results['gpu_fast_result'], 
                tokens_per_expert
            )
            
            if is_correct:
                print("Results match perfectly!")
            else:
                print("Results do not match!")
                
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()


def run():
    verify_expert_indices_result()


if __name__ == "__main__":
    run()