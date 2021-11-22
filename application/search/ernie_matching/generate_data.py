import numpy as np
import pandas as pd
import random
from tqdm import tqdm 
from sklearn.model_selection import train_test_split

def filterout_data(file_name,file_output):
    out=open(file_output,'w')
    with open(file_name) as f:
        for item in f.readlines():
            arr=item.strip().split('\t')
            if(len(arr)==2):
                out.write(item)
            else:
                print(item)
    out.close()


def  process_data(data_path,train_neg_num,test_neg_num):
    
    df=pd.read_csv(data_path,sep='\t',header=None)
    df=df.drop_duplicates()
    df.rename(columns={0:'query',1:'title'},inplace=True)
    train_data=df.sample(n=30000)
    # train_data['label']=train_data['title'].apply(lambda x:1)
    # train_data.to_csv('wanfang_train.csv',sep='\t')

    other_data=df[~df.index.isin(train_data.index)]

    test_data=df.sample(n=15000)
    other_data=other_data[~other_data.index.isin(test_data.index)]
    
    # generate train data
    list_train=train_data.values.tolist()
    list_data=[]
    for item in tqdm(list_train):
        if(type(item[0])==float):
            continue
        list_data.append([item[0],item[1],1])
    # print(df.head())
    other_data, train_neg = train_test_split(other_data, test_size=train_neg_num, random_state=42)
    list_extract = train_neg.values.tolist()
    other_data, test_neg = train_test_split(other_data, test_size=test_neg_num, random_state=42)
    
    sample_data=other_data['title'].tolist()
    for item in tqdm(list_extract):
        if(type(item[0])==float):
            continue
        item=[t.strip() for t in item]
        idx=random.randint(0, len(sample_data))
        if(sample_data[idx]!=item[1] and len(item[0])!=0):
            # print(item)
            list_data.append([item[0],sample_data[idx],0])
    neg_df=pd.DataFrame(list_data,columns=['query','title','label'])
    # df_igno_idx = pd.concat([train_data,neg_df], ignore_index=True)
    neg_df.to_csv('data/wanfang_train.csv',sep='\t',index=False)

    # generate test data
    list_test=test_data.values.tolist()
    list_test_data=[]
    for item in tqdm(list_test):
        if(type(item[0])==float):
            continue
        list_test_data.append([item[0],item[1],1])

    list_extract = test_neg.values.tolist()
    for item in tqdm(list_extract):
        if(type(item[0])==float):
            continue
        item=[t.strip() for t in item]
        idx=random.randint(0, len(sample_data))
        if(sample_data[idx]!=item[1] and len(item[0])!=0):
            # print(item)
            list_test_data.append([item[0],sample_data[idx],0])

    test_df=pd.DataFrame(list_test_data,columns=['query','title','label'])
    # df_igno_idx = pd.concat([train_data,neg_df], ignore_index=True)
    test_df.to_csv('data/wanfang_test.csv',sep='\t',index=False)

if __name__ == "__main__":
    data_path='train.csv'
    # train_neg_num=90000
    # test_neg_num=15000
    # process_data(data_path,train_neg_num,test_neg_num)
    
    train_neg_num=30000
    test_neg_num=15000
    process_data(data_path,train_neg_num,test_neg_num)
    # file_output='train_out.csv'
    # filterout_data(data_path,file_output)