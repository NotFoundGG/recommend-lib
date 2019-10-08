'''
@Author: Yu Di
@Date: 2019-09-29 10:54:50
@LastEditors: Yudi
@LastEditTime: 2019-10-08 23:52:50
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: User-KNN recommender
'''
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

from util.data_loader import load_rate
from util.metrics import ndcg_at_k, mean_average_precision, hr_at_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', 
                        type=int, 
                        default=10, 
                        help='top number of recommend list')
    parser.add_argument('--k', 
                        type=int, 
                        default=40, 
                        help='The (max) number of neighbors to take into account for aggregation')
    parser.add_argument('--mink', 
                        type=int, 
                        default=1, 
                        help='The minimum number of neighbors to take into account for aggregation')
    args = parser.parse_args()

    k = args.N

    df = load_rate('ml-100k')

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df=df[['user', 'item', 'rating']], reader=reader)
    train_set, test_set = train_test_split(data, test_size=.2)

    # params for item KNN
    sim_options = {'name': 'pearson_baseline', 'user_based': True}
    
    algo = KNNWithMeans(args.k, args.mink, sim_options=sim_options)
    algo.fit(train_set)

    test_set = pd.DataFrame(test_set, columns=['user', 'item', 'rating']) 
    # true item with some negative sampling items, compose 50 items as alternatives
    # count all items interacted in full dataset
    u_is = defaultdict(set)
    for _, row in df.iterrows():
        u_is[int(row['user'])].add(int(row['item']))

    test_u_is = defaultdict(set)
    for _, row in test_set.iterrows():
        test_u_is[int(row['user'])].add(int(row['item']))
    item_pool = test_set.item.unique().tolist()

    max_i_num = test_set.groupby('user')['item'].count().max()

    for key, val in test_u_is.items():
        cands_num = max_i_num - len(val)
        cands = np.random.choice(item_pool, cands_num).astype(int)
        test_u_is[key].update(set(cands))

    # get top-N list for test users
    preds = {}
    for u in test_u_is.keys():
        test_u_is[u] = list(test_u_is[u])
        pred_rates = [algo.predict(u, i)[0] for i in test_u_is[u]]
        rec_idx = np.argsort(pred_rates)[::-1][:k]
        top_n = np.array(test_u_is[u])[rec_idx]
        preds[u] = list(top_n)
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

    hr_k = hr_at_k(list(preds.values()))
    print(f'HR@{k}: {hr_k}')
