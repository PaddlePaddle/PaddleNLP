import pandas as pd

file_name='data/train.csv'
data=pd.read_csv(file_name,sep='\t')
print(data.shape)
data=data.drop_duplicates()
print(data.shape)

file_name='data/test.csv'
data=pd.read_csv(file_name,sep='\t',header=None)
print(data.shape)
data=data.drop_duplicates()
print(data.shape)