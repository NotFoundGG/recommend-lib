'''
@Author: Yu Di
@Date: 2019-09-29 10:54:40
@LastEditors: Yudi
@LastEditTime: 2019-09-29 16:04:06
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Popularity-based recommender
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
        # self.top_n = res[:self.N].index.tolist()
        self.rank_list = res.index.tolist()

    def predict(self, df):
        item_list = df['item'].unique().tolist()
        exists_item = [i for i in self.rank_list if i in item_list]
        non_exists = list(set(item_list) - set(exists_item))
        np.random.shuffle(non_exists)

        if len(exists_item) == 0:
            exists_item = []
        if len(non_exists) == 0:
            non_exists = []
        rank_list = exists_item + non_exists
        top_n = rank_list[:self.N]
        res = defaultdict(list)
        for u in df.user.unique():
            res[u] = top_n
        return res

if __name__ == '__main__':
    #TODO top-N setting
    k = 10

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
