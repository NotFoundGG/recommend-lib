'''
@Author: Yu Di
@Date: 2019-10-28 14:42:51
@LastEditors: Yudi
@LastEditTime: 2019-11-01 15:30:47
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: SVD recommender, also known as BiasMF
'''
import random
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

from util.matrix_factorization import SVD
from util.data_loader import load_rate
from util.metrics import ndcg_at_k, mean_average_precision, hr_at_k, precision_at_k, recall_at_k, mrr_at_k

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
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.005, 
                        help='The learning rate for all parameters')
    parser.add_argument('--reg', 
                        type=float, 
                        default=0.02,
                        help='The regularization term for all parameter')
    parser.add_argument('--topk', 
                        type=int, 
                        default=10, 
                        help='Number of recommendations')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    parser.add_argument('--data_split', 
                        type=str, 
                        default='fo', 
                        help='method for split test,options: loo/fo')
    parser.add_argument('--by_time', 
                        type=int, 
                        default=0, 
                        help='whether split data by time stamp')
    args = parser.parse_args()

    src = args.dataset
    df = load_rate(src)

    user_num, item_num = df.user.nunique(), df.item.nunique()
    df['user'] = pd.Categorical(df['user']).codes
    df['item'] = pd.Categorical(df['item']).codes

    if args.data_split == 'fo':
        if args.by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)

            split_idx = int(np.ceil(len(df) * 0.8))
            train_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()
        else:
            train_set, test_set = train_test_split(df, test_size=.2)
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
    
    algo = SVD(user_num, item_num, n_factors=args.factors, n_epochs=args.epochs, lr_all=args.lr, reg_all=args.reg)
    algo.fit(train_set)

    # true item with some negative sampling items, compose 50 items as alternatives
    # count all items interacted in full dataset
    u_is = defaultdict(set)
    for _, row in df.iterrows():
        u_is[int(row['user'])].add(int(row['item']))

    test_u_is = defaultdict(set)
    for _, row in test_set.iterrows():
        test_u_is[int(row['user'])].add(int(row['item']))
    item_pool = test_set.item.unique().tolist()

    max_i_num = 100
    for key, val in test_u_is.items():
        if len(val) < max_i_num:
            cands_num = max_i_num - len(val)
            # remove item appear in train set towards certain user
            sub_item_pool = [i for i in item_pool if i not in u_is[key]] 
            cands = random.sample(sub_item_pool, cands_num)
            test_u_is[key] = test_u_is[key] | set(cands)
        else:
            test_u_is[key] = random.sample(val, max_i_num)

    # get top-N list for test users
    preds = {}
    for u in test_u_is.keys():
        test_u_is[u] = list(test_u_is[u])
        pred_rates = [algo.predict(u, i) for i in test_u_is[u]]
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
    precision_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
    print(f'Precision@{args.topk}: {precision_k}')

    recall_k = np.mean([recall_at_k(r, len(ur[u]), args.topk) for u, r in preds.items()])
    print(f'Recall@{args.topk}: {recall_k}')

    map_k = mean_average_precision(list(preds.values()))
    print(f'MAP@{args.topk}: {map_k}')

    ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])
    print(f'NDCG@{args.topk}: {ndcg_k}')

    hr_k = hr_at_k(list(preds.values()))
    print(f'HR@{args.topk}: {hr_k}')

    mrr_k = mrr_at_k(list(preds.values()))
    print(f'MRR@{args.topk}: {mrr_k}')
