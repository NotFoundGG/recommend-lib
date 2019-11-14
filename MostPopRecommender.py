'''
@Author: Yu Di
@Date: 2019-09-29 10:54:40
@LastEditors: Yudi
@LastEditTime: 2019-11-14 10:49:05
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Popularity-based recommender
'''
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold

from util.data_loader import load_rate
from util.metrics import ndcg_at_k, map_at_k, hr_at_k, precision_at_k, recall_at_k, mrr_at_k

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
    parser.add_argument('--prepro', 
                        type=str, 
                        default='origin', 
                        help='dataset type for experiment, origin, 5core, 10core available')
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
    parser.add_argument('--val_method', 
                        type=str, 
                        default='cv', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    args = parser.parse_args()

    k = args.topk
    df = load_rate(args.dataset, args.prepro)
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

    train_set_list, val_set_list = [], []
    # train validation split
    if args.val_method == 'cv': # k-fold
        kf = KFold(n_splits=args.fold_num, shuffle=False, random_state=2019)
        for train_index, val_index in kf.split(train_set):
            train_set_list.append(train_set.iloc[train_index, :])
            val_set_list.append(train_set.iloc[val_index, :])
    elif args.val_method == 'loo':
        val_set = train_set.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        val_key = val_set[['user', 'item']].copy()
        train_set = train_set.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(val_key)).reset_index().copy()

        train_set_list.append(train_set)
        val_set_list.append(val_set)
    elif args.val_method == 'tloo':
        train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)

        train_set['rank_latest'] = train_set.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        new_train_set = train_set[train_set['rank_latest'] > 1].copy()
        val_set = train_set[train_set['rank_latest'] == 1].copy()
        del new_train_set['rank_latest'], val_set['rank_latest']

        train_set_list.append(new_train_set)
        val_set_list.append(val_set)
    elif args.val_method == 'tfo':
        train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)

        split_idx = int(np.ceil(len(train_set) * 0.9))
        train_set, val_set = train_set.iloc[:split_idx, :].copy(), train_set.iloc[split_idx:, :].copy()

        train_set_list.append(train_set)
        val_set_list.append(val_set)
    else:
        raise ValueError('Invalid val_method value, expect: cv, loo, tloo, tfo')

    fnl_precision, fnl_recall, fnl_map, fnl_ndcg, fnl_hr, fnl_mrr = [], [], [], [], [], []
    for fold in range(len(train_set_list)):
        print(f'Start train validation [{fold + 1}]')
        reco = MostPopRecommender(k)
        reco.fit(train_set_list[fold])
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
        fnl_precision.append(precision_k)

        recall_k = np.mean([recall_at_k(r, len(ur[u]), k) for u, r in preds.items()])
        fnl_recall.append(recall_k)

        map_k = map_at_k(list(preds.values()))
        fnl_map.append(map_k)

        ndcg_k = np.mean([ndcg_at_k(r, k) for r in preds.values()])
        fnl_ndcg.append(ndcg_k)

        hr_k = hr_at_k(list(preds.values()), list(preds.keys()), ur)
        fnl_hr.append(hr_k)

        mrr_k = mrr_at_k(list(preds.values()))
        fnl_mrr.append(mrr_k)

    print('---------------------------------')
    print(f'Precision@{args.topk}: {np.mean(fnl_precision)}')
    print(f'Recall@{args.topk}: {np.mean(fnl_recall)}')
    print(f'MAP@{args.topk}: {np.mean(fnl_map)}')
    print(f'NDCG@{args.topk}: {np.mean(fnl_ndcg)}')
    print(f'HR@{args.topk}: {np.mean(fnl_hr)}')
    print(f'MRR@{args.topk}: {np.mean(fnl_mrr)}')
