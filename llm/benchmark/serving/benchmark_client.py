import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import List, Tuple
import numpy as np
import requests
from sentencepiece import SentencePieceProcessor
from typing import AsyncGenerator, List, Optional, Tuple, Union
import argparse


def infer(
    session_id: int,
    req_que: mp.Queue,
    res_que: mp.Queue,
    end_flags: mp.Queue,
    backend: str = "tgi",
    api_url: str = "http://localhost:8010/generate_stream",
    model_name: str = "openlm-research/open_llama_13b",
    min_dec_len: int = 1,
    max_dec_len: int = 2048
):
    stats = []
    while not req_que.empty():
        try:
            prompt, input_seqlen, output_seqlen = req_que.get(timeout=10.0)
        except:
            continue

        start = time.time()
        is_first = True
        first_token_latency = float("inf")
        # if output_seqlen > 1024:
        #     print("Request exceeds 1024 tokens. Truncating to 1024.", output_seqlen)
        #     output_seqlen = 1024

        headers = {"User-Agent": "Benchmark Client"}
        if backend == "vllm":
            pload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": output_seqlen,
                "stream": True
            }
        elif backend == "sglang":
            pload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_dec_len,
                    "temperature": 0.6,
                    "top_k": 1,
                    "top_p": 0.95,
                },
                "return_logprob": False,
                "stream": True
            }
        elif backend == "paddle":
            pload = {
                "text": prompt,
                "max_dec_len": max_dec_len,
                "min_dec_len": min_dec_len,
                "topp": 0.95,
                "temperature": 0.6,
                "stream": True,
                "return_all_tokens": False
            }
        else:  # backend is trtllm
            pload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": max_dec_len,
                "stream": True
            }

        response = requests.post(url=api_url, headers=headers, json=pload, stream=True)

        chunks = []
        for chunk in response.iter_content(chunk_size=1000000):
            chunks.append(chunk)
            if is_first:
                first_token_latency = time.time() - start
                is_first = False

        total_token_latency = time.time() - start

        if backend == "vllm":
            try:
                res_text = b"".join(chunks).decode("utf-8")
                token_num = eval(
                    res_text.split("previous_num_tokens:")[-1].split("data: [DONE]")[0]
                )
            except:
                token_num = len(chunks)
        elif backend == "paddle":
            token_num = 0
            for chunk in chunks:
                chunk_dict = json.loads(chunk.decode("utf-8").strip())
                if chunk_dict["is_end"]:
                    token_num += chunk_dict["tokens_all_num"]
        else:
            token_num = len(chunks)

        stats.append(
            [first_token_latency, total_token_latency, input_seqlen, output_seqlen, token_num]
        )
        print(f"Request queue size: {req_que.qsize()}, Real return tokens: {token_num}, Request Chunks: {len(chunks)}, Label out_seq_len: {output_seqlen}")

    print(f"Process ID {os.getpid()} has processed all requests.")
    if len(stats) > 0:
        res_que.put((session_id, stats))
        end_flags.put(1)
    print(f"Session {session_id} (PID={os.getpid()}) completed.")


def warmup(
    concurrency: int,
    output_seqlen: int,
    warmup_round: int = 4,
    backend: str = "tgi",
    api_url: str = "http://localhost:8010/generate_stream",
    model_name: str = "openlm-research/open_llama_13b",
):
    print("Starting warmup process...")

    def _infer(index, warmup_round, backend):
        headers = {"User-Agent": "Benchmark Client"}
        if backend == "vllm":
            pload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "temperature": 1.0,
                "top_p": 0.0,
                "stream": True,
            }
        elif backend == "sglang":
            pload = {
                "text": "how about beijing?",
                "sampling_params": {
                    "max_new_tokens": output_seqlen,
                    "temperature": 1.0,
                    "top_k": 1,
                    "top_p": 0.0,
                },
                "return_logprob": False,
                "stream": True
            }
        elif backend == "paddle":
            pload = {
                "text": "how about beijing?",
                "max_dec_len": output_seqlen,
                "topp": 1.0,
                "temperature": 1.0,
                "stream": True,
                "return_all_tokens": False
            }
        else:  # trtllm
            pload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 50,
                "stream": True
            }

        for _ in range(warmup_round):
            response = requests.post(url=api_url, headers=headers, json=pload, stream=True)
            for _ in response.iter_content(chunk_size=1024):
                pass

    start_time = time.perf_counter()
    procs = []
    for i in range(concurrency):
        proc = mp.Process(target=_infer, args=(i, warmup_round, backend))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    end_time = time.perf_counter()
    print(f"Warmup completed. Elapsed time: {round(end_time - start_time, 2)} seconds")


def sample_requests_filtered_shared_gpt(dataset_path: str, num_requests: int):
    with open(dataset_path, "r") as infile:
        filtered_dataset = json.load(infile)
    extracted_data = [[entry[0], entry[1], entry[2]] for entry in filtered_dataset[:num_requests]]
    que = mp.Queue()

    total_input_len = sum(entry[1] for entry in extracted_data)
    total_output_len = sum(entry[2] for entry in extracted_data)
    avg_input_len = total_input_len / len(extracted_data)
    avg_output_len = total_output_len / len(extracted_data)

    print(f"Number of test samples: {len(extracted_data)}")
    print(f"Average input length: {avg_input_len:.2f}")
    print(f"Average output length: {avg_output_len:.2f}")

    for data in extracted_data:
        que.put(data)
    print(f"Total samples added to request queue: {len(extracted_data)}")
    return que


def sample_requests_inner(dataset_path: str, num_requests: int):
    start_time = time.perf_counter()
    dataset = []
    prompts_token_lens = []
    completions_token_lens = []
    text_test = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            prompts_token_lens.append(int(data["input_token_num"]))
            completions_token_lens.append(int(data["min_dec_len"]) - 1)
            dataset.append([data["input_ids"], ""])
            text_test.append(data["text"])

    print(
        f"Input length range: [{min(prompts_token_lens)}, {max(prompts_token_lens)}], "
        f"Output length range: [{min(completions_token_lens)}, {max(completions_token_lens)}]"
    )
    avg_input_len = sum(prompts_token_lens) / len(prompts_token_lens)
    avg_output_len = sum(completions_token_lens) / len(completions_token_lens)

    print(f"Average input length: {avg_input_len:.2f}")
    print(f"Average output length: {avg_output_len:.2f}")
    print(f"Elapsed time for tokenization: {round(time.perf_counter() - start_time, 2)} seconds")

    start_time = time.perf_counter()
    filtered_dataset = []
    for (prompt, _), input_len, output_len, text in zip(
        dataset, prompts_token_lens, completions_token_lens, text_test
    ):
        filtered_dataset.append([text, input_len, output_len])

    sampled_requests = random.sample(filtered_dataset, num_requests)
    print(f"Number of sampled requests: {len(sampled_requests)}")
    que = mp.Queue()
    for data in sampled_requests:
        que.put(data)
    print(f"Elapsed time for filtering: {round(time.perf_counter() - start_time, 2)} seconds")
    return que

def save_results_to_file(stats: np.ndarray, output_file: str):
    """Save the benchmark results to a file."""
    results = {
        "first_token_latency": {
            "min": stats[:, 0].min(),
            "max": stats[:, 0].max(),
            "avg": stats[:, 0].mean(),
        },
        "total_token_latency": {
            "min": stats[:, 1].min(),
            "max": stats[:, 1].max(),
            "avg": stats[:, 1].mean(),
        },
        "input_sequence_length": {
            "min": stats[:, 2].min(),
            "max": stats[:, 2].max(),
            "avg": stats[:, 2].mean(),
        },
        "output_sequence_length": {
            "min": stats[:, 3].min(),
            "max": stats[:, 3].max(),
            "avg": stats[:, 3].mean(),
        },
        "real_output_sequence_length": {
            "min": stats[:, 4].min(),
            "max": stats[:, 4].max(),
            "avg": stats[:, 4].mean(),
        },
        "qps": len(stats) / (stats[:, 1].sum() / len(stats)),
        "real_output_tokens_per_second": len(stats) / (stats[:, 1].sum() / len(stats)) * stats[:, 4].mean(),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")


def main(
    dataset_path: str = "./filtered_sharedgpt_short_3000.json",
    concurrency: int = 256,
    num_prompts: int = 3000,
    backend: str = "paddle",
    model_name: str = "openlm-research/open_llama_13b",
    host: str = "localhost",
    port: str = "8100",
    warmup_round: int = 1,
    dataset_name: str = "sharegpt",
    min_dec_len: int = 1,
    max_dec_len: int = 2048,
    output_file: Optional[str] = None,
):
    api_url = f"http://{host}:{port}/generate_stream"
    if backend in ["vllm", "trtllm", "paddle"]:
        api_url = f"http://{host}:{port}/v1/chat/completions"
    elif backend == "sglang":
        api_url = f"http://{host}:{port}/generate"
    else:
        raise RuntimeError("Unsupported backend. Choose from: paddle, vllm, trtllm, sglang")

    print(f"API URL: {api_url}")
    warmup(concurrency, 256, warmup_round, backend, api_url, model_name)
    print(f"Dataset name: {dataset_name}")

    if dataset_name == "sharegpt":
        req_que = sample_requests_filtered_shared_gpt(dataset_path, num_prompts)
    elif dataset_name == "paddle_inner":
        req_que = sample_requests_inner(dataset_path, num_prompts)
    else:
        raise ValueError("Invalid dataset name. Choose from: sharegpt, paddle_inner")

    res_que = mp.Queue()
    procs = []
    end_flags = mp.Queue()
    start_time = time.perf_counter()
    for i in range(concurrency):
        proc = mp.Process(
            target=infer,
            args=(i + 1, req_que, res_que, end_flags, backend, api_url, model_name, min_dec_len, max_dec_len),
        )
        procs.append(proc)
        proc.start()

    while end_flags.qsize() < concurrency:
        time.sleep(0.01)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    stats = [[]] * (concurrency + 1)
    while res_que.qsize() > 0:
        session_id, _stats = res_que.get()
        stats[session_id] = np.array(_stats)

    all_stat = []
    for i in range(concurrency):
        print(f"Session {i + 1} processed {len(stats[i + 1])} requests.")
        if len(stats[i + 1]) > 0:
            all_stat.append(stats[i + 1])

    all_stat = np.concatenate(all_stat).reshape(-1, 5)

    print(f"Processed {len(all_stat)} requests in {elapsed_time:.2f} seconds.")
    print(
        f"First Token Latency (min, max, avg): {all_stat[:, 0].min():.2f}, {all_stat[:, 0].max():.2f}, {all_stat[:, 0].mean():.2f}"
    )
    print(
        f"Total Token Latency (min, max, avg): {all_stat[:, 1].min():.2f}, {all_stat[:, 1].max():.2f}, {all_stat[:, 1].mean():.2f}"
    )
    print(
        f"Input Sequence Length (min, max, avg): {all_stat[:, 2].min():.2f}, {all_stat[:, 2].max():.2f}, {all_stat[:, 2].mean():.2f}"
    )
    print(
        f"Output Sequence Length (min, max, avg): {all_stat[:, 3].min():.2f}, {all_stat[:, 3].max():.2f}, {all_stat[:, 3].mean():.2f}"
    )
    print(
        f"Real Output Sequence Length (min, max, avg): {all_stat[:, 4].min():.2f}, {all_stat[:, 4].max():.2f}, {all_stat[:, 4].mean():.2f}"
    )
    print(f"QPS: {len(all_stat) / elapsed_time:.2f}")
    print(f"Real Output Tokens/s: {len(all_stat) / elapsed_time * all_stat[:, 4].mean():.2f}")

    # Save results to file if output_file is provided
    if output_file:
        save_results_to_file(all_stat, output_file)

    for p in procs:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script for inference performance.")
    parser.add_argument("--dataset_path", type=str, default="./filtered_sharedgpt_short_3000.json", help="Path to the dataset file.")
    parser.add_argument("--concurrency", type=int, default=256, help="Number of concurrent requests.")
    parser.add_argument("--num_prompts", type=int, default=3000, help="Number of prompts to process.")
    parser.add_argument("--backend", type=str, default="paddle", choices=["paddle", "vllm", "trtllm", "sglang"], help="Backend to use for inference.")
    parser.add_argument("--model_name", type=str, default="openlm-research/open_llama_13b", help="Name of the model to use.")
    parser.add_argument("--host", type=str, default="localhost", help="Host address of the inference server.")
    parser.add_argument("--port", type=str, default="8100", help="Port of the inference server.")
    parser.add_argument("--warmup_round", type=int, default=1, help="Number of warmup rounds.")
    parser.add_argument("--dataset_name", type=str, default="sharegpt", choices=["sharegpt", "paddle_inner"], help="Name of the dataset to use.")
    parser.add_argument("--min_dec_len", type=int, default=1, help="Minimum decoding length.")
    parser.add_argument("--max_dec_len", type=int, default=2048, help="Maximum decoding length.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the results file.")

    args = parser.parse_args()
    main(
        dataset_path=args.dataset_path,
        concurrency=args.concurrency,
        num_prompts=args.num_prompts,
        backend=args.backend,
        model_name=args.model_name,
        host=args.host,
        port=args.port,
        warmup_round=args.warmup_round,
        dataset_name=args.dataset_name,
        min_dec_len=args.min_dec_len,
        max_dec_len=args.max_dec_len,
        output_file=args.output_file,
    )