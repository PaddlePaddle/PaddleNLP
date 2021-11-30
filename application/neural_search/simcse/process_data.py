import pandas as pd
from tqdm import tqdm 

def merge_columns():
    root_path='data/train.csv'
    file=open('data/log.csv','w')
    clean_file=open('data/train_unsupervised.csv','w')
    with open(root_path,'r') as f:
        for line in tqdm(f.readlines()):
            linearr=line.strip().split('\t')
            for item in linearr:
                clean_file.write(item+'\n')

merge_columns()
# file_name='data/train.csv'
# data=pd.read_csv(file_name,sep='\t')
# print(data.head())