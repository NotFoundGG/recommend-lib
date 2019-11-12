'''
@Author: Yu Di
@Date: 2019-10-30 13:52:23
@LastEditors: Yudi
@LastEditTime: 2019-11-12 15:31:09
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Pure SVD
'''
import os
import random
import argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
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
                        default=1, 
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

    src = args.dataset
    dataset = WRMFData(src, data_split=args.data_split, by_time=args.by_time, 
                       val_method=args.val_method, fold_num=args.fold_num)

    # calculate metrics
    print(f'Start Calculating KPI metrics, validation method: {args.val_method}......')
    val_kpi = []
    fnl_precision, fnl_recall, fnl_map, fnl_ndcg, fnl_hr, fnl_mrr = [], [], [], [], [], []
    for fold in range(len(dataset.train_list)):
        assert min(dataset.train_list[fold].shape) >= args.factors, 'Invalid sigular value number, must be less than the minimum of matrix shape'
        u, s, vh = sp.linalg.svds(dataset.train_list[fold].asfptype(), args.factors)
        smat = np.diag(s)
        predict_mat = u.dot(smat.dot(vh))

        print(f'Start validation [{fold + 1}]......')
        # generate top-N list for validation user set
        val_user_set = dataset.val_users_list[fold]
        ur = defaultdict(list)
        index = dataset.val.nonzero()
        for u, i in zip(index[0], index[1]):
            ur[u].append(i)
        candidates = defaultdict(list)
        for u in val_user_set:
            unint = np.where(dataset.train_list[fold][u, :].toarray().reshape(-1) == 0)[0]
            candidates[u] = list(set(unint) & set(ur[u]))
            # [i for i in unint if i in ur[u]]
        max_i_num = 100
        preds = {}
        # item_pool = list(set(dataset.val.nonzero()[1]))
        item_pool = list(range(dataset.item_num))
        for u in tqdm(val_user_set):
            if len(candidates[u]) < max_i_num:
                actual_cands = set(candidates[u])
                neg_item_pool = [i for i in range(dataset.train_list[fold].shape[1]) if i not in ur[u]]
                neg_cands = random.sample(neg_item_pool, max_i_num - len(candidates[u])) 
                cands = actual_cands | set(neg_cands)
            else:
                cands = random.sample(candidates[u], max_i_num)
            pred_rates = predict_mat[u, list(cands)]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            preds[u] = list(np.array(list(cands))[rec_idx])
        for u in preds.keys():
            preds[u] = [1 if i in ur[u] else 0 for i in preds[u]]

        val_kpi_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
        val_kpi.append(val_kpi_k)

        # genereate top-N list for test user set
        test_user_set = dataset.test_users
        ur = defaultdict(list) # actually interacted items by user u
        index = dataset.test.nonzero()
        for u, i in zip(index[0], index[1]):
            ur[u].append(i)
        candidates = defaultdict(list)
        for u in test_user_set:
            unint = np.where(dataset.train_list[fold][u, :].toarray().reshape(-1) == 0)[0] # 未交互的物品
            candidates[u] = list(set(unint) & set(ur[u]))# 未交互的物品中属于后续已交互的物品

        max_i_num = 100
        preds = {}
        # item_pool = list(set(dataset.test.nonzero()[1]))
        item_pool = list(range(dataset.item_num))
        for u in tqdm(test_user_set):
            if len(candidates[u]) < max_i_num:
                actual_cands = set(candidates[u])
                neg_item_pool = [i for i in range(dataset.train_list[fold].shape[1]) if i not in ur[u]]
                neg_cands = random.sample(neg_item_pool, max_i_num - len(candidates[u])) 
                cands = actual_cands | set(neg_cands)
            else:
                cands = random.sample(candidates[u], max_i_num)
            pred_rates = predict_mat[u, list(cands)]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            preds[u] = list(np.array(list(cands))[rec_idx])
        for u in preds.keys():
            preds[u] = [1 if i in ur[u] else 0 for i in preds[u]]

        precision_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
        fnl_precision.append(precision_k)

        recall_k = np.mean([recall_at_k(r, len(ur[u]), args.topk) for u, r in preds.items()])
        fnl_recall.append(recall_k)

        map_k = map_at_k(list(preds.values()))
        fnl_map.append(map_k)

        ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])
        fnl_ndcg.append(ndcg_k)

        hr_k = hr_at_k(list(preds.values()), list(preds.keys()), ur)
        fnl_hr.append(hr_k)

        mrr_k = mrr_at_k(list(preds.values()))
        fnl_mrr.append(mrr_k)

    for i in range(len(val_kpi)):
        print(f'Validation [{i + 1}] Precision@{args.topk}: {val_kpi[i]}')

    print('---------------------------------')
    print(f'Precision@{args.topk}: {np.mean(fnl_precision)}')
    print(f'Recall@{args.topk}: {np.mean(fnl_recall)}')
    print(f'MAP@{args.topk}: {np.mean(fnl_map)}')
    print(f'NDCG@{args.topk}: {np.mean(fnl_ndcg)}')
    print(f'HR@{args.topk}: {np.mean(fnl_hr)}')
    print(f'MRR@{args.topk}: {np.mean(fnl_mrr)}')
