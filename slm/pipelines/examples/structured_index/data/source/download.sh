#!/bin/bash

download() {
    local url=$1
    local ext=$2

    # 获取文件的basename
    local filename=$(basename "$url")

    # 检查文件名是否以ext结尾
    if [[ "$filename" != *".$ext" ]]; then
        filename="$filename.$ext"
    fi

    # 下载文件
    echo "Downloading $url as $filename"
    curl -o "$filename" "$url"
}

# 读取JSON文件并解析所有ext和对应的URL
json_file="source_url.json"
exts=$(jq -r 'keys[]' "$json_file")

# 遍历每个ext并下载对应的文件
for ext in $exts; do
    urls=$(jq -r --arg ext "$ext" '.[$ext][]' "$json_file")
    for url in $urls; do
        download "$url" "$ext"
    done
done