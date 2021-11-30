import pandas as pd


def generate_recall_dataset(num):

    file_name='data/train.csv'
    train_data=pd.read_csv(file_name,sep='\t',header=None)
    data=train_data.sample(n=num,random_state=123)
    data.rename(columns={0:'query',1:'title'},inplace=True)
    # print(data.head())

    file_name='data/test.csv'
    test_data=pd.read_csv(file_name,sep='\t',header=None)
    test_data.rename(columns={0:'query',1:'title'},inplace=True)
    # print(test_data.head())

    df=pd.concat([data,test_data],axis=0,ignore_index=True)
    # print(df.head())
    # print(df.columns)
    df.to_csv('data/corpus_{}.csv'.format(num),sep='\t',columns=['title'],index=False,header=None)

def generate_milvus_dataset():

    file_name='data/train.csv'
    train_data=pd.read_csv(file_name,sep='\t',header=None)
    # data=train_data.sample(n=num,random_state=123)
    data=train_data
    data.rename(columns={0:'query',1:'title'},inplace=True)
    # print(data.head())

    file_name='data/test.csv'
    test_data=pd.read_csv(file_name,sep='\t',header=None)
    test_data.rename(columns={0:'query',1:'title'},inplace=True)
    # print(test_data.head())

    df=pd.concat([data,test_data],axis=0,ignore_index=True)
    # print(df.head())
    # print(df.columns)
    df.to_csv('data/milvus_corpus.csv',sep='\t',columns=['title'],index=False,header=None)

def merge_corups(list_paths,output_path):
    file=open(output_path,'w')
    num=10000000
    count=0
    for file_path in list_paths:
        with open(file_path) as f:
            for item in f.readlines():
                count+=1
                file.write(item)
                if(count>=num):
                    break
    file.close()



if __name__ == "__main__":
    # num=580000
    # generate_recall_dataset(num)
    # generate_milvus_dataset()
    list_paths=['data/milvus_corpus.csv','data/other_files.txt']
    output_path='data/milvus_data.csv'
    merge_corups(list_paths,output_path)