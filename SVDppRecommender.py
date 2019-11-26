'''
@Author: Yu Di
@Date: 2019-10-28 14:42:51
@LastEditors: Yudi
@LastEditTime: 2019-11-26 14:36:38
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: SVDpp recommender, also known as SVD++
'''
import gc
import random
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold

from util.matrix_factorization import SVDpp
from util.data_loader import load_rate
from util.metrics import ndcg_at_k, map_at_k, hr_at_k, precision_at_k, recall_at_k, mrr_at_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepro', 
                        type=str, 
                        default='origin', 
                        help='dataset type for experiment, origin, 5core, 10core available')
    parser.add_argument('--factors', 
                        type=int, 
                        default=20, 
                        help='The number of latent factors')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help='The number of iteration of the SGD procedure')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.007, 
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
    parser.add_argument('--val_method', 
                        type=str, 
                        default='tfo', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    args = parser.parse_args()

    df = load_rate(args.dataset, args.prepro)

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

    train_set_list, val_set_list = [], []
    # train validation split
    if args.val_method == 'cv': # 5-fold
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

    algo_list = []
    for i in range(len(train_set_list)):
        print(f'Start train model with fold {i + 1}')
        algo = SVDpp(user_num, item_num, n_factors=args.factors, n_epochs=args.epochs, lr_all=args.lr, reg_all=args.reg)
        algo.fit(train_set)
        algo_list.append(algo)

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
            sub_item_pool = set(item_pool) - set(u_is[key])
            cands = random.sample(sub_item_pool, cands_num)
            test_u_is[key] = test_u_is[key] | set(cands)
        else:
            test_u_is[key] = random.sample(val, max_i_num)

    print('---------------------------------')
    print('Start Calculating KPI......')
    # k-fold consider
    val_kpi = []
    fnl_precision, fnl_recall, fnl_map, fnl_ndcg, fnl_hr, fnl_mrr = [], [], [], [], [], []
    for i in tqdm(range(len(algo_list))):
        algo = algo_list[i]
        # get top-N list for validation users
        # validation ground truth setting
        val_set = val_set_list[i]
        val_u_is = defaultdict(set)
        for _, row in val_set.iterrows():
            val_u_is[int(row['user'])].add(int(row['item']))
        # validation set candidates setting
        max_i_num = 100
        for key, val in val_u_is.items():
            if len(val) < max_i_num:
                cands_num = max_i_num - len(val)
                cands = random.sample(item_pool, cands_num)
                val_u_is[key] = val_u_is[key] | set(cands)
            else:
                val_u_is[key] = random.sample(val, max_i_num)
        preds = {}
        for u in val_u_is.keys():
            val_u_is[u] = list(val_u_is[u])
            pred_rates = [algo.predict(u, i) for i in val_u_is[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(val_u_is[u])[rec_idx]
            preds[u] = list(top_n)
        # get actual interaction info. of validation users
        ur = defaultdict(list)
        for u in val_set.user.unique():
            ur[u] = val_set.loc[val_set.user==u, 'item'].values.tolist()
        for u in preds.keys():
            preds[u] = [1 if e in ur[u] else 0 for e in preds[u]]
        val_kpi_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
        val_kpi.append(val_kpi_k)

        # get top-N list for test users
        preds = {}
        for u in test_u_is.keys():
            test_u_is[u] = list(test_u_is[u])
            pred_rates = [algo.predict(u, i) for i in test_u_is[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(test_u_is[u])[rec_idx]
            preds[u] = list(top_n)
        # get actual interaction info. of test users
        test_ur = defaultdict(list)
        for u in test_set.user.unique():
            test_ur[u] = test_set.loc[test_set.user==u, 'item'].values.tolist()
        for u in preds.keys():
            preds[u] = [1 if e in test_ur[u] else 0 for e in preds[u]]

        # calculate metrics
        precision_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
        fnl_precision.append(precision_k)

        recall_k = np.mean([recall_at_k(r, len(test_ur[u]), args.topk) for u, r in preds.items()])
        fnl_recall.append(recall_k)

        map_k = map_at_k(list(preds.values()))
        fnl_map.append(map_k)

        ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])
        fnl_ndcg.append(ndcg_k)

        hr_k = hr_at_k(list(preds.values()), list(preds.keys()), test_ur)
        fnl_hr.append(hr_k)
        
        mrr_k = mrr_at_k(list(preds.values()))
        fnl_mrr.append(mrr_k) 

        gc.collect()

    for i in range(len(val_kpi)):
        print(f'Validation [{i + 1}] Precision@{args.topk}: {val_kpi[i]}')

    print('---------------------------------')
    print(f'Precision@{args.topk}: {np.mean(fnl_precision)}')
    print(f'Recall@{args.topk}: {np.mean(fnl_recall)}')
    print(f'MAP@{args.topk}: {np.mean(fnl_map)}')
    print(f'NDCG@{args.topk}: {np.mean(fnl_ndcg)}')
    print(f'HR@{args.topk}: {np.mean(fnl_hr)}')
    print(f'MRR@{args.topk}: {np.mean(fnl_mrr)}')
