'''
@Author: Yu Di
@Date: 2019-10-28 14:42:51
@LastEditors: Yudi
@LastEditTime: 2019-10-28 15:05:31
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: NMF recommender
'''
import random
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import NMF
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

from util.data_loader import load_rate
from util.metrics import ndcg_at_k, mean_average_precision, hr_at_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--factors', 
                        type=int, 
                        default=100, 
                        help='The number of latent factors')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help='The number of iteration of the SGD procedure')
    parser.add_argument('--reg_u', 
                        type=float, 
                        default=0.06,
                        help='The regularization term for user factors')
    parser.add_argument('--reg_i', 
                        type=float, 
                        default=0.06,
                        help='The regularization term for item factors')
    parser.add_argument('--topk', 
                        type=int, 
                        default=10, 
                        help='Number of recommendations')

    args = parser.parse_args()

    src = 'ml-100k'
    df = load_rate(src)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df=df[['user', 'item', 'rating']], reader=reader)
    train_set, test_set = train_test_split(data, test_size=.2)
    
    algo = NMF(n_factors=args.factors, n_epochs=args.epochs, reg_pu=args.reg_u, reg_qi=args.reg_i)
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
    max_i_num = 50 if max_i_num <= 50 else max_i_num

    for key, val in test_u_is.items():
        cands_num = max_i_num - len(val)
        # cands = np.random.choice(item_pool, cands_num).astype(int)
        cands = random.sample(item_pool, cands_num)
        test_u_is[key] = test_u_is[key] | set(cands)
    
    # get top-N list for test users
    preds = {}
    for u in test_u_is.keys():
        test_u_is[u] = list(test_u_is[u])
        pred_rates = [algo.predict(u, i)[0] for i in test_u_is[u]]
        rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
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
    print(f'MAP@{args.topk}: {map_k}')

    ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])
    print(f'NDCG@{args.topk}: {ndcg_k}')

    hr_k = hr_at_k(list(preds.values()))
    print(f'HR@{args.topk}: {hr_k}')
