'''
@Author: Yu Di
@Date: 2019-10-27 19:13:22
@LastEditors: Yudi
@LastEditTime: 2019-10-28 14:34:33
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: SLIM recommender
'''
import os
import time
import pickle
import numbers
import argparse
import operator

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.linear_model import SGDRegressor, ElasticNet
from concurrent.futures import ProcessPoolExecutor

from util import slim
from util.data_loader import SlimData
from util.metrics import mean_average_precision, ndcg_at_k, hr_at_k

class SLIM(object):
    def __init__(self, data):
        self.data = data
        print('Start SLIM recommendation......')
        self.A = self.__user_item_matrix()
        self.alpha = None
        self.lam_bda = None
        self.max_iter = None
        self.tol = None # learning threshold
        self.N = None # top-N num
        self.lambda_is_ratio = None

        self.W = None
        self.recommendation = None
    
    def __user_item_matrix(self):
        A = np.zeros((self.data.num_user, self.data.num_item))
        for user, item in self.data.train:
            A[user, item] = 1
        return A

    def __aggregation_coefficients(self):
        group_size = 100  # 并行计算每组计算的行/列数
        n = self.data.num_item // group_size  # 并行计算分组个数
        starts = []
        ends = []
        for i in range(n):
            start = i * group_size
            starts.append(start)
            ends.append(start + group_size)
        if self.data.num_item % group_size != 0:
            starts.append(n * group_size)
            ends.append(self.data.num_item)
            n += 1

        print('covariance updates pre-calculating')
        covariance_array = None
        with ProcessPoolExecutor() as executor:
            covariance_array = np.vstack(executor.map(slim.compute_covariance, [self.A] * n, starts, ends))

        slim.symmetrize_covariance(covariance_array)

        print('coordinate descent for learning W matrix......')
        if self.lambda_is_ratio:
            with ProcessPoolExecutor() as executor:
                return np.hstack(executor.map(slim.coordinate_descent_lambda_ratio, 
                                              [self.alpha] * n, 
                                              [self.lam_bda] * n, 
                                              [self.max_iter] * n, 
                                              [self.tol] * n, 
                                              [self.data.num_user] * n, 
                                              [self.data.num_item] * n, 
                                              [covariance_array] * n, 
                                              starts, ends))
        else:
            with ProcessPoolExecutor() as executor:
                return np.hstack(executor.map(slim.coordinate_descent, 
                                              [self.alpha] * n, 
                                              [self.lam_bda] * n, 
                                              [self.max_iter] * n, 
                                              [self.tol] * n, 
                                              [self.data.num_user] * n, 
                                              [self.data.num_item] * n, 
                                              [covariance_array] * n, 
                                              starts, ends))
    
    def __recommend(self, user_AW, user_item_set):
        '''
        generate N recommend items for user
        :param user_AW: the user row of the result of matrix dot product of A and W
        :param user_item_set: item interacted in train set for user 
        :return: recommend list for user
        '''
        rank = dict()
        for i in set(range(self.data.num_item)) - user_item_set:
            rank[i] = user_AW[i]
        return [items[0] for items in sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:self.N]]

    def __get_recommendation(self):
        train_user_items = [set() for u in range(self.data.num_user)]
        for user, item in self.data.train:
            train_user_items[user].add(item)

        AW = self.A.dot(self.W)

        # recommend N items for each user
        recommendation = []
        for user_AW, user_item_set in zip(AW, train_user_items):
            recommendation.append(self.__recommend(user_AW, user_item_set))
        return recommendation

    def compute_recommendation(self, alpha=0.5, lam_bda=0.02, max_iter=1000, tol=0.0001, N=10, lambda_is_ratio=True):
        self.alpha = alpha
        self.lam_bda = lam_bda
        self.max_iter = max_iter
        self.tol = tol
        self.N = N
        self.lambda_is_ratio = lambda_is_ratio

        print(f'Start calculating W matrix(alpha={self.alpha}, lambda={self.lam_bda}, max_iter={self.max_iter}, tol={self.tol})')
        self.W = self.__aggregation_coefficients()

        print(f'Start calculating recommendation list(N={self.N})')
        self.recommendation = self.__get_recommendation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', 
                        type=int, 
                        default=10, 
                        help='top number of recommend list')
    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.5, 
                        help='ratio if lasso result, 0 for ridge-regression, 1 for lasso-regression')
    parser.add_argument('--elastic', 
                        type=float, 
                        default=0.02, 
                        help='elastic net parameter')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=1000, 
                        help='No. of learning iteration')
    parser.add_argument('--tol', 
                        type=float, 
                        default=0.0001, 
                        help='learning threshold')
    args = parser.parse_args()

    src = 'ml-100k'
    slim_data= SlimData(src)

    start_time = time.time()
    recommend = SLIM(slim_data)
    recommend.compute_recommendation(alpha=args.alpha, lam_bda=args.elastic, 
                                     max_iter=args.epochs, tol=args.tol, N=args.topk)
    print('Finish train model and generate topN list')

    # genereate top-N list for test user set
    test_user_set = list({ele[0] for ele in slim_data.test})
    ur = defaultdict(list)
    for ele in slim_data.test:
        ur[ele[0]].append(ele[1])
    preds = {}
    for u in ur.keys():
        preds[u] = recommend.recommendation[u]
    # kpi calculation
    for u in preds.keys():
        preds[u] = [1 if e in ur[u] else 0 for e in preds[u]]

    # calculate metrics
    map_k = mean_average_precision(list(preds.values()))
    print(f'MAP@{args.topk}: {map_k}')

    ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])
    print(f'NDCG@{args.topk}: {ndcg_k}')

    hr_k = hr_at_k(list(preds.values()))
    print(f'HR@{args.topk}: {hr_k}')