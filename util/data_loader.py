'''
@Author: Yu Di
@Date: 2019-09-29 11:10:53
@LastEditors: Yudi
@LastEditTime: 2019-09-29 11:16:45
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import pandas as pd

def load_rate(src='ml100k'):
    df = pd.read_csv('./data/ml-100k/u.data', sep='\t', header=None, 
                     names=['user', 'item', 'rating', 'timestamp'], engine='python')

    return df
