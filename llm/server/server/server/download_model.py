import requests
import os
from tqdm import tqdm
import argparse
import hashlib
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="download models")
    parser.add_argument('-m', '--model_name', default='deepseek-ai/DeepSeek-R1/weight_only_int4',
                       help="model_name")
    parser.add_argument('-d', '--dir', default='downloads',
                       help="save dir")
    parser.add_argument('-n', '--nnodes', type=int, default=1,
                       help="the number of node")
    parser.add_argument('-M', '--mode', default="master", choices=["master", "slave"],
                       help="only support in 2 nodes model. There are two modes, master or slave.")
    parser.add_argument('-s', '--speculate_model_path', default=None,
                       help="speculate model path")
    return parser.parse_args()

def calculate_md5(file_path, chunk_size=8192):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_file(url, save_path, md5sum):
    """download file"""
    md5_check= int(os.getenv("MD5_CHECK", 0)) == 1
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            if os.path.exists(save_path):
                if not md5_check:
                    print(f"{save_path} already exists and md5 check is off, skip this step")
                    return save_path
                current_md5sum = calculate_md5(save_path)
                if md5sum != current_md5sum:
                    os.remove(save_path)
                    print("not complete file! start to download again")
                else:
                    print(f"{save_path} already exists and md5sum matches")
                    return save_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            file_name = save_path.split('/')[-1]
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(
                total=total_size, 
                unit='iB', 
                unit_scale=True,
                desc=f"download {file_name}"
            )

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            if total_size != 0 and os.path.getsize(save_path) != total_size:
                raise RuntimeError("not complete")
                
            return save_path
    except Exception as e:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)
        return None

def download_from_txt(base_url, save_dir):
    txt_url = base_url + "/file_list.txt"
    print(f"{txt_url}")
    try:
        response = requests.get(txt_url)
        response.raise_for_status()
        files_name = response.text.splitlines()
        files_name  = [file.strip() for file in files_name if file.strip()]

        md5sum = [file_name.rsplit(':', 1)[-1] for file_name in files_name]
        file_name = [file_name.rsplit(':', 1)[0] for file_name in files_name]

        if not files_name:
            print("No valid files found.")
            return

        print(f"Found {len(files_name)} files")

        for i in range(len(file_name)): 
            cur_url = base_url + f"/{file_name[i]}"
            path = download_file(cur_url, os.path.join(save_dir, file_name[i]), md5sum[i])
            if path:
                print(f"[✓] Success: {path}")
            else:
                print(f"[×] Failed: {cur_url}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download file list from {txt_url}: {str(e)}")

def main():
    args = parse_arguments()
    print(f"Save Path: {os.path.abspath(args.dir)}")

    # make dir
    path = os.path.join(args.dir, args.model_name)
    os.makedirs(path, exist_ok=True)

    model_name = args.model_name
    env = os.environ
    # Define supported model patterns
    supported_patterns = [
        r".*Qwen.*", 
        r".+Llama.+",
        r".+Mixtral.+", 
        r".+DeepSeek.+",
    ]
    
    # Check if model_name matches any supported pattern
    if not any(re.match(pattern, model_name) for pattern in supported_patterns):
        raise ValueError(
            f"{model_name} is not in the supported list. Currently supported models: Qwen, Llama, Mixtral, DeepSeek. Please check the model name from this document https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md"
        )
    print(f"Start downloading model: {model_name}")
    tag = env.get("tag", "3.0.0.b4")
    base_url = f"https://paddlenlp.bj.bcebos.com/models/static/{tag}/{model_name}"
    temp_file = None
    if args.nnodes == 1:
        temp_file = "model"
    elif args.nnodes > 1:
        if args.mode == "master":
            temp_file = "node1"
        elif args.mode == "slave":
            temp_file = "node2"
        else:
            raise ValueError(f"Invalid mode: {args.mode}. Mode must be 'master' or 'slave'.")
    else:
        raise ValueError(f"Invalid nnodes: {args.nnodes}. nnodes must be >= 1.")

    if temp_file:
        model_url = base_url + f"/{temp_file}"
        download_from_txt(model_url, path)
    else:
        print(f"Don't support download the {model_name} in mode {args.mode}")

    if args.speculate_model_path:
        os.makedirs(args.speculate_model_path, exist_ok=True)
        print(f"Start downloading mtp model: {model_name}")
        model_url = base_url + "/mtp"
        download_from_txt(model_url, args.speculate_model_path)

if __name__ == "__main__":
    main()
