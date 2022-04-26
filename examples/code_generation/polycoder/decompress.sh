#!/bin/bash

n=0
maxjobs=2  # 最大进程数
m=0
maxfiles=12800  # 每个目录中的最大文件数

for i in $(ls openwebtext); do 
    echo $i; 
    if  ((n % $maxfiles == 0)); then
        ((m=n))
        mkdir -p raw_data/data_$m
    fi
    if  ((++n % $maxjobs == 0)) ; then
        wait 
    fi
    tar xJf openwebtext/$i --warning=no-timestamp -C raw_data/data_$m/ &
done
