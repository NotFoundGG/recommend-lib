'''
@Author: Yu Di
@Date: 2019-10-30 13:52:23
@LastEditors: Yudi
@LastEditTime: 2019-11-11 10:12:30
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Pure SVD
'''
import os
import random
import argparse
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

from util.data_loader import load_rate, WRMFData
from util.metrics import hr_at_k, ndcg_at_k, map_at_k, precision_at_k, recall_at_k, mrr_at_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', 
                        type=int, 
                        default=10, 
                        help='recommend number for rank list')
    parser.add_argument('--factors', 
                        type=int, 
                        default=150, 
                        help='No. of singular value preserved')
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
    args = parser.parse_args()

    src = args.dataset
    dataset = WRMFData(src, data_split=args.data_split, by_time=args.by_time)

    assert min(dataset.train.shape) >= args.factors, 'Invalid sigular value number, must be less than the minimum of matrix shape'
    u, s, vh = sp.linalg.svds(dataset.train.asfptype(), args.factors)
    smat = np.diag(s)
    predict_mat = u.dot(smat.dot(vh))

    # genereate top-N list for test user set
    test_user_set = dataset.test_users
    ur = defaultdict(list) # actually interacted items by user u
    index = dataset.test.nonzero()
    for u, i in zip(index[0], index[1]):
        ur[u].append(i)
    candidates = defaultdict(list)
    for u in test_user_set:
        unint = np.where(dataset.train[u, :].toarray().reshape(-1) == 0)[0] # 未交互的物品
        candidates[u] = [i for i in unint if i in ur[u]] # 未交互的物品中属于后续已交互的物品

    max_i_num = 100
    preds = {}
    item_pool = list(set(dataset.test.nonzero()[1]))
    for u in test_user_set:
        if len(candidates[u]) < max_i_num:
            actual_cands = set(candidates[u])
            neg_item_pool = [i for i in range(dataset.train.shape[1]) if i not in ur[u]]
            neg_cands = random.sample(neg_item_pool, max_i_num - len(candidates[u])) 
            cands = actual_cands | set(neg_cands)
        else:
            cands = random.sample(candidates[u], max_i_num)
        pred_rates = predict_mat[u, list(cands)]
        rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
        preds[u] = list(np.array(list(cands))[rec_idx])
    for u in preds.keys():
        preds[u] = [1 if i in ur[u] else 0 for i in preds[u]]

    # calculate metrics
    print('Start Calculating KPI metrics......')
    precision_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
    print(f'Precision@{args.topk}: {precision_k}')

    recall_k = np.mean([recall_at_k(r, len(ur[u]), args.topk) for u, r in preds.items()])
    print(f'Recall@{args.topk}: {recall_k}')

    map_k = map_at_k(list(preds.values()))
    print(f'MAP@{args.topk}: {map_k}')

    ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])
    print(f'NDCG@{args.topk}: {ndcg_k}')

    hr_k = hr_at_k(list(preds.values()))
    print(f'HR@{args.topk}: {hr_k}')

    mrr_k = mrr_at_k(list(preds.values()))
    print(f'MRR@{args.topk}: {mrr_k}')
