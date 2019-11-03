'''
@Author: Yu Di
@Date: 2019-09-29 10:54:40
@LastEditors: Yudi
@LastEditTime: 2019-11-01 15:12:47
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Popularity-based recommender
'''
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

from util.data_loader import load_rate
from util.metrics import ndcg_at_k, mean_average_precision, hr_at_k, precision_at_k, recall_at_k, mrr_at_k

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', 
                        type=int, 
                        default=10, 
                        help='top number of recommend list')
    parser.add_argument('--data_split', 
                        type=str, 
                        default='fo', 
                        help='method for split test,options: loo/fo')
    parser.add_argument('--by_time', 
                        type=int, 
                        default=0, 
                        help='whether split data by time stamp')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    args = parser.parse_args()

    k = args.topk
    src = args.dataset
    df = load_rate(src)
    # split dataset
    if args.data_split == 'fo':
        if args.by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)
            split_idx = int(np.ceil(len(df) * 0.8))
            train_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()
        else:
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=2019)
    elif args.data_split == 'loo':
        if args.by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)
            df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
            train_set, test_set = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
            del train_set['rank_latest'], test_set['rank_latest']
        else:
            test_set = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
            test_key = test_set[['user', 'item']].copy()
            train_set = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()
    else:
        raise ValueError('Invalid data_split value, expect: loo, fo')

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
    precision_k = np.mean([precision_at_k(r, k) for r in preds.values()])
    print(f'Precision@{k}: {precision_k}')

    recall_k = np.mean([recall_at_k(r, len(ur[u]), k) for u, r in preds.items()])
    print(f'Recall@{k}: {recall_k}')

    map_k = mean_average_precision(list(preds.values()))
    print(f'MAP@{k}: {map_k}')

    ndcg_k = np.mean([ndcg_at_k(r, k) for r in preds.values()])
    print(f'NDCG@{k}: {ndcg_k}')

    hr_k = hr_at_k(list(preds.values()))
    print(f'HR@{k}: {hr_k}')

    mrr_k = mrr_at_k(list(preds.values()))
    print(f'MRR@{k}: {mrr_k}')
