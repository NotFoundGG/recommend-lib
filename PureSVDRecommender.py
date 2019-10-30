'''
@Author: Yu Di
@Date: 2019-10-30 13:52:23
@LastEditors: Yudi
@LastEditTime: 2019-10-30 14:48:27
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Pure SVD
'''
import os
import random
import argparse
import numpy as np
from collections import defaultdict

from util.data_loader import load_rate, WRMFData
from util.metrics import hr_at_k, ndcg_at_k, mean_average_precision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', 
                        type=int, 
                        default=10, 
                        help='recommend number for rank list')
    args = parser.parse_args()

    src = 'ml-100k'
    dataset = WRMFData(src)

    u, s, vh = np.linalg.svd(dataset.train.A, full_matrices=False)
    smat = np.diag(s)
    predict_mat = np.dot(u, np.dot(smat, vh))

    # genereate top-N list for test user set
    test_user_set = dataset.test_users
    ur = defaultdict(list) # u的实际交互item
    index = dataset.test.nonzero()
    for u, i in zip(index[0], index[1]):
        ur[u].append(i)
    candidates = defaultdict(list)
    for u in test_user_set:
        unint = np.where(dataset.train[u, :].toarray().reshape(-1) == 0)[0] # 未交互的物品
        candidates[u] = [i for i in unint if i in ur[u]] # 未交互的物品中属于后续已交互的物品

    max_i_num = max([len(v) for v in candidates.values()])
    max_i_num = 50 if max_i_num <= 50 else max_i_num
    preds = {}
    item_pool = list(set(dataset.test.nonzero()[1]))
    for u in test_user_set:
        actual_cands = set(candidates[u])
        neg_item_pool = [i for i in range(dataset.train.shape[1]) if i not in ur[u]]
        neg_cands = random.sample(neg_item_pool, max_i_num - len(candidates[u])) 
        cands = actual_cands | set(neg_cands)
        pred_rates = predict_mat[u, list(cands)]
        rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
        preds[u] = list(np.array(list(cands))[rec_idx])
    for u in preds.keys():
        preds[u] = [1 if i in ur[u] else 0 for i in preds[u]]

    # calculate metrics
    print('Start Calculating KPI metrics......')
    map_k = mean_average_precision(list(preds.values()))
    print(f'MAP@{args.topk}: {map_k}')

    ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])
    print(f'NDCG@{args.topk}: {ndcg_k}')

    hr_k = hr_at_k(list(preds.values()))
    print(f'HR@{args.topk}: {hr_k}')