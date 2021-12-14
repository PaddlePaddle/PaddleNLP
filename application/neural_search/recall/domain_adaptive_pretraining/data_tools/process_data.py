import pandas as pd
from sklearn.model_selection import train_test_split

def create_pretraining_data():
    file_name='data/wanfang_text.csv'
    data=pd.read_csv(file_name,sep='\t')
    data=data.drop_duplicates()
    print(data.shape)
    data.to_csv('data/pretrain_data.csv',sep='\t',index=False)

def process_data():
    ouput=open('wanfangdata/wanfang_text.txt','w')
    with open('data/pretrain_data.csv') as f:
        for i,item in enumerate(f.readlines()):
            if(i==0):
                continue
            arr=item.strip().split('\t')
            # queryText
            ouput.write(arr[-2]+'\n')
            ouput.write('\n')
            # title
            ouput.write(arr[-1]+'\n')
            # abstract
            ouput.write(arr[-3]+'\n')
            ouput.write('\n')


if __name__=="__main__":
    create_pretraining_data()
    process_data()
