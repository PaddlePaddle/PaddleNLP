import json
import os
import requests
import time
from typing import List, Tuple
from tqdm import tqdm
import argparse

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

def download_and_cache_file(url: str, filename: str) -> str:
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename
    try:
        print(f"Downloading from {url} to {filename}")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024  # Download in chunks of 1KB

        with open(filename, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    except Exception as e:
        print(f"Network error, please download from huggingface : {SHAREGPT_URL}")
        return None

    return filename

def filter_shared_gpt(
    ori_shared_dataset_path: str,
    tokenizer_name: str,
    num_prompts: int,
    backend: str,
):
    start = time.perf_counter()

    if backend == "paddle":
        from paddlenlp.transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("please install transformers when using other backends")
            return
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )

    dataset = []
    prompts_token_lens = []
    completions_token_lens = []
    with open(ori_shared_dataset_path) as f:
        dataset_json = json.load(f)

    dataset = [data for data in dataset_json if len(data["conversations"]) >= 2]

    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
    filtered_dataset: List[Tuple[str, int, int]] = []

    for i in range(min(num_prompts, len(dataset))):
        if len(filtered_dataset) == num_prompts:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids)
        if prompt_len < 2 or output_len < 2:
            continue

        if prompt_len > 1024 or prompt_len + output_len > 2048:
            continue

        prompts_token_lens.append(prompt_len)
        completions_token_lens.append(output_len)
        filtered_dataset.append([prompt, prompt_len, output_len])

    averge_input_len = sum(prompts_token_lens) / len(prompts_token_lens)
    averge_output_len = sum(completions_token_lens) / len(completions_token_lens)
    print(f"filtered averge_input_len: {averge_input_len}; averge_output_len: {averge_output_len}")
    filtered_dataset_path = f"filtered_sharedgpt_short_{num_prompts}.json"
    with open(filtered_dataset_path, "w") as f:
        json.dump(filtered_dataset, f, indent=4)
    print(f"filtered dataset save done ! Averge_input_len: {averge_input_len}, Averge_output_len: {averge_output_len} .")

def donw_and_filter_dataset(tokenizer_name, num_prompts, backend, dataset_path: str = "./ShareGPT_V3_unfiltered_cleaned_split.json"):
    if not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL, dataset_path)
        if dataset_path is None:
            return
    filter_shared_gpt(dataset_path, tokenizer_name, num_prompts, backend)

def main(
    tokenizer_name: str,
    dataset_path: str = "./ShareGPT_V3_unfiltered_cleaned_split.json",
    num_prompts: int = 3000,
    backend: str = "paddle",
):
    donw_and_filter_dataset(tokenizer_name, num_prompts, backend, dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and filter ShareGPT dataset.")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Name of the tokenizer to use.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=3000,
        help="Number of prompts to filter.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle",
        help="Backend to use for tokenizer (paddle or transformers).",
    )

    args = parser.parse_args()
    main(
        tokenizer_name=args.tokenizer_name,
        dataset_path=args.dataset_path,
        num_prompts=args.num_prompts,
        backend=args.backend,
    )