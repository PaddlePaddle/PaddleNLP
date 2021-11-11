import pandas as pd
from tqdm import tqdm 
import csv

def merge_text(x):
    if(type(x['keywords'])==float):
        return x['article_title']
    elif(type(x['article_title'])==float):
        return ""
    else:
        return x['article_title']+','.join(x['keywords'].split('%'))
    

def merge_keywords_title():

    # root_path='./data/part-00000-bb25d15e-d079-46a3-bd2f-c2b5bea8f164-c000.csv'
    root_path='data/clean_data.csv'
    data=pd.read_csv(root_path,sep='\t',error_bad_lines=False,quoting=csv.QUOTE_NONE)
    # data['keywords']=data['keywords'].apply(lambda x:str(x).replace('\n','').replace('\t',''))
    # data['article_title']=data['article_title'].apply(lambda x:str(x).replace('\n','').replace('\t',''))

    # data['keywords']=data['keywords'].apply(lambda x:','.join(str(x).split('%')))
    # data['title']=data['article_title']+data['keywords']
    # data['title']=data['title'].apply(lambda x:x.replace('"',""))
    data['title']=data.apply(lambda x:merge_text(x),axis=1)
    print(data.shape)

    data.to_csv('data/wanfang_clean.csv',columns=['queryText','title'],sep='\t',index=False)

def clean_data():
    root_path='./data/part-00000-bb25d15e-d079-46a3-bd2f-c2b5bea8f164-c000.csv'
    file=open('data/log.csv','w')
    clean_file=open('data/clean_data.csv','w')
    with open(root_path,'r') as f:
        for line in tqdm(f.readlines()):
            linearr=line.strip().split('\t')
            if(len(linearr)!=4):
                file.write(line)
            else:
                line=line.replace('"',"")
                clean_file.write(line.strip().replace('\n',"")+'\n')

def output_example():
    root_path='./data/part-00000-bb25d15e-d079-46a3-bd2f-c2b5bea8f164-c000.csv'
    clean_file=open('data/example.csv','w')
    with open(root_path,'r') as f:
        for idx,line in enumerate(tqdm(f.readlines())):
            clean_file.write(line)
            if(idx>3):
                break

def get_line(idx):
    root_path='data/clean_data.csv'
    with open(root_path,'r') as f:
        data_arr=f.readlines()
        line=data_arr[idx]
        print(line)
        print(line.split())

def extract_ernie_data(data_path,output_path):
    file=open(output_path,'w')
    with open(data_path) as f:
        for item in f.readlines():
            if(len(item)>3):
                file.write(item)
    file.close()

def merge_files(data_path,output_path):
    file=open(output_path,'a')
    with open(data_path) as f:
        for item in f.readlines():
            if(len(item)>3):
                file.write(item)
    file.close()



if __name__=="__main__":
    # clean_data()
    # merge_keywords_title()
    # output_example()
    # idx=47
    # get_line(idx)
    num=10
    for i in tqdm(range(num)):
        data_path='./data/ernie_1.0/data_%02d' %(i)
        print(data_path)
        output_path='./data/ernie_processed/ernie_data_%2d.txt' %(i)
        extract_ernie_data(data_path,output_path)
        corpus_path='./data/other_files.txt'
        merge_files(output_path,corpus_path)