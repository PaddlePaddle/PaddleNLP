import requests
import random
import json
from hashlib import md5
from tqdm import tqdm 
import time

# Set your own appid/appkey.
appid = ''
appkey = ''


# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
def convert_cn_to_en(from_lang,to_lang,query):
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    print(result)
    return result["trans_result"]

from_lang = 'zh'
to_lang =  'en'

def generate_insurance():

    file_path='data/baoxian/dev.csv'
    file=open('data/baoxian/test_pair.csv','w')
    with open(file_path,'r') as f:
        for i,line in tqdm(enumerate(f.readlines())):
            time.sleep(1)
            test_str=line.strip()
            res=convert_cn_to_en(from_lang,to_lang,test_str)
            time.sleep(1)
            eng_text=res[0]['dst']
            res=convert_cn_to_en(to_lang,from_lang,eng_text)
            zh_s=res[0]['dst']
            file.write(test_str+'\t'+zh_s+'\n')

    file.close()

generate_insurance()