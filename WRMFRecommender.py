'''
@Author: Yu Di
@Date: 2019-10-28 15:47:50
@LastEditors: Yudi
@LastEditTime: 2019-10-30 13:21:03
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
from util.metrics import hr_at_k, ndcg_at_k, mean_average_precision

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
    args = parser.parse_args()

    src = 'ml-100k'
    dataset = WRMFData(src)

    algo = WRMF(dataset.train, lambda_val=args.lambda_val, alpha=args.alpha, 
                iterations=args.epochs, factor_num=args.factors)
    algo.fit()

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
        neg_item_pool = [i for i in range(algo.num_item) if i not in ur[u]]
        neg_cands = random.sample(neg_item_pool, max_i_num - len(candidates[u])) 
        cands = actual_cands | set(neg_cands)
        pred_rates = algo.user_vec[u, :].dot(algo.item_vec).toarray()[0, list(cands)]
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
