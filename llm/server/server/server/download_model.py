import requests
import os
from tqdm import tqdm
import argparse
import hashlib

def parse_arguments():
    parser = argparse.ArgumentParser(description="download models")
    parser.add_argument('-u', '--url', required=True,
                        help="downloadfiles")
    parser.add_argument('-d', '--dir', default='downloads',
                       help="save dir")
    parser.add_argument('-m', '--model_name', default='file_list.txt',
                       help="model_name")

    return parser.parse_args()


def calculate_md5(file_path, chunk_size=8192):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_file(url, save_path, md5sum):
    """download file"""

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            if os.path.exists(save_path):
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

        md5sum = [file_name.split(':')[-1] for file_name in files_name]
        file_name = [file_name.split(':')[0] for file_name in files_name]

        if not files_name:
            print("No valid files found.")
            return

        print(f"Found {len(files_name)} files")


        for i in range(len(file_name)): 
            cur_url = base_url + f"/{file_name[i]}"
            path = download_file(cur_url, save_dir+f"/{file_name[i]}", md5sum[i])
            if path:
                print(f"[✓] Success: {path}")
            else:
                print(f"[×] Failed: {cur_url}")
    except Exception as e:
        raise Exception(f"Failed to get model {file_name} from {base_url}, please recheck the model name from https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md")


def main():
    args = parse_arguments()
    print(f"Save Path: {os.path.abspath(args.dir)}")

    # make dir
    os.makedirs(args.dir, exist_ok=True)

    # download from txt
    download_from_txt(args.url, args.dir, args.model_name)

if __name__ == "__main__":
    main()



