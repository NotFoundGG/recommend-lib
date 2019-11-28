'''
@Author: Yu Di
@Date: 2019-10-28 15:47:50
@LastEditors: Yudi
@LastEditTime: 2019-11-28 16:19:06
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: WRMF
'''
import os
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from util.data_loader import load_rate, WRMFData
from util.metrics import hr_at_k, ndcg_at_k, map_at_k, precision_at_k, recall_at_k, mrr_at_k

class WRMF(object):
    def __init__(self, train_set, lambda_val=0.1, alpha=40, iterations=10, factor_num=20, seed=2019):
        self.epochs = iterations
        self.rstate = np.random.RandomState(seed)
        self.C = alpha * train_set
        self.num_user, self.num_item = self.C.shape[0], self.C.shape[1]

        self.X = sp.csr_matrix(self.rstate.normal(scale=0.01, size=(self.num_user, factor_num)))
        self.Y = sp.csr_matrix(self.rstate.normal(scale=0.01, size=(self.num_item, factor_num)))
        self.X_eye = sp.eye(self.num_user)
        self.Y_eye = sp.eye(self.num_item)
        self.lambda_eye = lambda_val * sp.eye(factor_num)

    def fit(self):
        for _ in tqdm(range(self.epochs)):
            yTy = self.Y.T.dot(self.Y)
            xTx = self.X.T.dot(self.X)
            for u in range(self.num_user):
                Cu = self.C[u, :].toarray()
                Pu = Cu.copy()
                Pu[Pu != 0] = 1
                CuI = sp.diags(Cu, [0])
                yTCuIY = self.Y.T.dot(CuI).dot(self.Y)
                yTCuPu = self.Y.T.dot(CuI + self.Y_eye).dot(Pu.T)
                self.X[u] = spsolve(yTy + yTCuIY + self.lambda_eye, yTCuPu)
            for i in range(self.num_item):
                Ci = self.C[:, i].T.toarray()
                Pi = Ci.copy()
                Pi[Pi != 0] = 1
                CiI = sp.diags(Ci, [0])
                xTCiIX = self.X.T.dot(CiI).dot(self.X)
                xTCiPi = self.X.T.dot(CiI + self.X_eye).dot(Pi.T)
                self.Y[i] = spsolve(xTx + xTCiIX + self.lambda_eye, xTCiPi)

        self.user_vec, self.item_vec = self.X, self.Y.T
    
    def predict(self, u, i):
        prediction = self.user_vec[u, :].dot(self.item_vec[:, i])
        return prediction.A[0, 0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepro', 
                        type=str, 
                        default='origin', 
                        help='dataset type for experiment, origin, 5core, 10core available')
    parser.add_argument('--lambda_val', 
                        type=float, 
                        default=0.1, 
                        help='regularization for ALS')
    parser.add_argument('--alpha', 
                        type=float, 
                        default=40, 
                        help='confidence weight')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=30, 
                        help='No. of training epochs')
    parser.add_argument('--factors', 
                        type=int, 
                        default=20, 
                        help='latent factor number')
    parser.add_argument('--topk', 
                        type=int, 
                        default=10, 
                        help='recommend number for rank list')
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

    dataset = WRMFData(args.dataset, data_split=args.data_split, by_time=args.by_time, 
                       val_method=args.val_method, fold_num=args.fold_num, prepro=args.prepro)

    print(f'Start Calculating KPI metrics, validation method: {args.val_method}......')
    val_kpi = []
    fnl_precision, fnl_recall, fnl_map, fnl_ndcg, fnl_hr, fnl_mrr = [], [], [], [], [], []
    for fold in range(len(dataset.train_list)):
        print(f'Start train validation [{fold + 1}]......')

        algo = WRMF(dataset.train_list[fold], lambda_val=args.lambda_val, alpha=args.alpha, 
                    iterations=args.epochs, factor_num=args.factors)
        algo.fit()

        print(f'Start validation [{fold + 1}] kpi calculation......')
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
        max_i_num = 1000
        preds = {}
        item_pool = list(range(dataset.item_num))
        for u in tqdm(val_user_set):
            if len(candidates[u]) < max_i_num:
                actual_cands = set(candidates[u])
                neg_item_pool = set(range(dataset.train_list[fold].shape[1])) - set(ur[u])
                neg_cands = random.sample(neg_item_pool, max_i_num - len(candidates[u])) 
                cands = actual_cands | set(neg_cands)
            else:
                cands = random.sample(candidates[u], max_i_num)
            pred_rates = algo.user_vec[u, :].dot(algo.item_vec).toarray()[0, list(cands)]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            preds[u] = list(np.array(list(cands))[rec_idx])
        for u in preds.keys():
            preds[u] = [1 if i in ur[u] else 0 for i in preds[u]]

        val_kpi_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
        val_kpi.append(val_kpi_k)

        print('Start test kpi calculation......')
        # genereate top-N list for test user set
        test_user_set = dataset.test_users
        test_ur = defaultdict(list) # u的实际交互item
        index = dataset.test.nonzero()
        for u, i in zip(index[0], index[1]):
            test_ur[u].append(i)
        candidates = defaultdict(list)
        for u in test_user_set:
            unint = np.where(dataset.train_list[fold][u, :].toarray().reshape(-1) == 0)[0] # 未交互的物品
            candidates[u] = list(set(unint) & set(test_ur[u])) # 未交互的物品中属于后续已交互的物品

        max_i_num = 1000
        preds = {}
        item_pool = list(range(dataset.item_num))
        for u in tqdm(test_user_set):
            if len(candidates[u]) < max_i_num:
                actual_cands = set(candidates[u])
                neg_item_pool = set(item_pool) - set(test_ur[u]) - set(ur[u])
                neg_cands = random.sample(neg_item_pool, max_i_num - len(candidates[u])) 
                cands = actual_cands | set(neg_cands)
            else:
                cands = random.sample(candidates[u], max_i_num)
            pred_rates = algo.user_vec[u, :].dot(algo.item_vec).toarray()[0, list(cands)]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            preds[u] = list(np.array(list(cands))[rec_idx])
        for u in preds.keys():
            preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]
    
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

    for i in range(len(val_kpi)):
        print(f'Validation [{i + 1}] Precision@{args.topk}: {val_kpi[i]}')

    print('---------------------------------')
    print(f'Precision@{args.topk}: {np.mean(fnl_precision)}')
    print(f'Recall@{args.topk}: {np.mean(fnl_recall)}')
    print(f'MAP@{args.topk}: {np.mean(fnl_map)}')
    print(f'NDCG@{args.topk}: {np.mean(fnl_ndcg)}')
    print(f'HR@{args.topk}: {np.mean(fnl_hr)}')
    print(f'MRR@{args.topk}: {np.mean(fnl_mrr)}')
