# coding = utf-8

import csv
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import random
csv.field_size_limit(sys.maxsize)

data_1 = pd.read_csv('/search/hadoop02/suanfa/kinghuangdd/text_similarity/train_test/train.csv',
                   encoding= 'utf-8',sep='\t')
data_2 = pd.read_csv('/search/hadoop02/suanfa/kinghuangdd/text_similarity/train_test/test.csv',
                   encoding= 'utf-8',sep='\t')

#data = pd.concat([data_1,data_2],axis=0)
data_1.append(data_2,sort=True )
data = data_1

train_examples,test_examples = train_test_split(data, test_size=0.05)

#dev_examples = test_examples

train_examples.to_csv('train.csv',encoding='utf-8',index=False,sep='\t')
#dev_examples.to_csv('dev.csv',encoding='utf-8',index=False)
test_examples.to_csv('test.csv',encoding='utf-8',index=False,sep='\t')
