import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_train_test_base():
    file_name='data/wanfang_clean.csv'
    data=pd.read_csv(file_name,sep='\t')
    data=data.drop_duplicates()
    print(data.shape)
    train,test=train_test_split(data,test_size=20000)
    print(test.shape)
    print(train.shape)

    train.to_csv('data/train.csv',sep='\t',index=False,header=None)
    test.to_csv('data/test.csv',sep='\t',index=False,header=None)

def split_train_ratio(ratio_or_size):
    file_name='data/train.csv'
    data=pd.read_csv(file_name,sep='\t')
    print(data.shape)
    train,test=train_test_split(data,test_size=ratio_or_size,random_state=2021)
    print(test.shape)
    print(train.shape)
    base_path='data/train_{}'.format(ratio_or_size)
    os.makedirs(base_path,exist_ok=True)
    train.to_csv(base_path+'/train_unsupervise.csv',sep='\t',index=False,header=None)
    test.to_csv(base_path+'/train.csv',sep='\t',index=False,header=None)


if __name__ == "__main__":
    split_train_test_base()
    # ratio_or_size=0.1
    # split_train_ratio(ratio_or_size)
    # ratio_or_size=0.01
    # split_train_ratio(ratio_or_size)
    ratio_or_size=0.001
    split_train_ratio(ratio_or_size)
    # ratio_or_size=0.0001
    # split_train_ratio(ratio_or_size)