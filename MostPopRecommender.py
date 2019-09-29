'''
@Author: Yu Di
@Date: 2019-09-29 10:54:40
@LastEditors: Yudi
@LastEditTime: 2019-09-29 14:56:55
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

from data_loader import load_rate
from metrics import ndcg_at_k, mean_average_precision

class MostPopRecommender(object):
    def __init__(self, N=5):
        self.N = N

    def fit(self, df):
        '''most popular item'''
        res = df['item'].value_counts()
        self.top_n = res[:self.N].index.tolist()

    def predict(self, df):
        res = defaultdict(list)
        for u in df.user.unique():
            res[u] = self.top_n
        return res

if __name__ == '__main__':
    #TODO top-N setting
    k = 5

    df = load_rate('ml100k')
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=2019)

    reco = MostPopRecommender(k)
    reco.fit(train_set)

    # get top-N list for test users
    preds = reco.predict(test_set)
    # get actual interaction info. of test users
    ur = defaultdict(list)
    for u in test_set.user.unique():
        ur[u] = test_set.loc[test_set.user==u, 'item'].values.tolist()
    for u in preds.keys():
        preds[u] = [1 if e in ur[u] else 0 for e in preds[u]]

    # calculate metrics
    map_k = mean_average_precision(list(preds.values()))
    print(f'MAP@{k}: {map_k}')

    ndcg_k = np.mean([ndcg_at_k(r, k) for r in preds.values()])
    print(f'NDCG@{k}: {ndcg_k}')
