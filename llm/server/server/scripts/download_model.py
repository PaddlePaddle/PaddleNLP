import requests
import os
from tqdm import tqdm
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="download models")
    parser.add_argument('-u', '--url', required=True,
                        help="downloadfiles")
    parser.add_argument('-d', '--dir', default='downloads',
                       help="save dir")
    parser.add_argument('-m', '--model_name', default='DeepSeek-R1/DeepSeek-R1-Distill-Llama-8B/weight_only_int8',
                       help="model_name")

    return parser.parse_args()


def download_from_txt(base_url, save_dir, model_name):
    txt_url = base_url + "/file_list.txt"
    try:
        response = requests.get(txt_url)
        response.raise_for_status()
        files_name = response.text.splitlines()
        files_name  = [file.strip() for file in files_name if file.strip()]

        if not files_name:
            print("No valid files found.")
            return

        print(f"Found {len(files_name)} files")

        for file_name in files_name:
            cur_url = base_url + f"/{file_name}"
            path = download_file(cur_url, save_dir)
            if path:
                print(f"[✓] Success: {path}")
            else:
                print(f"[×] Failed: {cur_url}")

    except Exception as e:
        print(f"Failed to get model {model_name}, please recheck the model name from https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md")
def download_file(url, save_path):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            file_name = url.split('/')[-1]
            save_path = os.path.join(save_path, file_name)
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

def main():
    args = parse_arguments()
    print(f"Save Path: {os.path.abspath(args.dir)}")

    # create save dir
    os.makedirs(args.dir, exist_ok=True)

    # download from file_list.txt
    download_from_txt(args.url, args.dir, args.model_name)

if __name__ == "__main__":
    main()