'''
@Author: Yu Di
@Date: 2019-09-29 10:54:50
@LastEditors: Yudi
@LastEditTime: 2019-11-01 15:29:46
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: User-KNN recommender
'''
import random
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

from util.data_loader import load_rate
from util.metrics import ndcg_at_k, mean_average_precision, hr_at_k, precision_at_k, recall_at_k, mrr_at_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', 
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
    parser.add_argument('--sim_method', 
                        type=str, 
                        default='pearson', 
                        help='method to calculate similarity, options for cosine, msd, pearson')
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

    if args.data_split == 'fo':
        if args.by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)
            df['user'] = pd.Categorical(df['user']).codes
            df['item'] = pd.Categorical(df['item']).codes
            split_idx = int(np.ceil(len(df) * 0.8))
            train_df, test_df = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()
            train_df = train_df.sort_values(['user', 'item']).reset_index(drop=True)
            test_set = []
            for _, row in test_df.iterrows():
                test_set.append((row['user'], row['item'], row['rating']))
            reader = Reader(rating_scale=(1, 5))
            train_set = Dataset.load_from_df(df=train_df[['user', 'item', 'rating']], reader=reader)
            train_set = train_set.build_full_trainset()
        else:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df=df[['user', 'item', 'rating']], reader=reader)
            train_set, test_set = train_test_split(data, test_size=.2)
    elif args.data_split == 'loo':
        if args.by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)
            df['user'] = pd.Categorical(df['user']).codes
            df['item'] = pd.Categorical(df['item']).codes
            df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
            train_df, test_df = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
            train_df = train_df.sort_values(['user', 'item']).reset_index(drop=True)
            del train_df['rank_latest'], test_df['rank_latest']
            test_set = []
            for _, row in test_df.iterrows():
                test_set.append((row['user'], row['item'], row['rating']))
            reader = Reader(rating_scale=(1, 5))
            train_set = Dataset.load_from_df(df=train_df[['user', 'item', 'rating']], reader=reader)
            train_set = train_set.build_full_trainset()
        else:
            df['user'] = pd.Categorical(df['user']).codes
            df['item'] = pd.Categorical(df['item']).codes
            test_df = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
            test_key = test_df[['user', 'item']].copy()
            train_df = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()
            test_set = []
            for _, row in test_df.iterrows():
                test_set.append((row['user'], row['item'], row['rating']))
            reader = Reader(rating_scale=(1, 5))
            train_set = Dataset.load_from_df(df=train_df[['user', 'item', 'rating']], reader=reader)
            train_set = train_set.build_full_trainset()
    else:
        raise ValueError('Invalid data_split value, expect: loo, fo')

    # params for item KNN
    sim_options = {'name': args.sim_method, 'user_based': True}
    
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
