'''
@Author: Yu Di
@Date: 2019-10-28 15:47:50
@LastEditors: Yudi
@LastEditTime: 2019-10-29 17:59:28
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: WRMF
'''
import os
import random
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from util.data_loader import load_rate, WRMFData

class WRMF(object):
    def __init__(self, train_set, lambda_val=0.1, alpha=40, iterations=10, factor_num=20, seed=2019):
        self.epochs = iterations
        self.rstate = np.random.RandomState(seed)
        self.C = alpha * train_set
        self.num_user, self.num_item = self.C.shape[0], self.C.shape[1]

        self.X = sp.csr_matrix(self.rstate.normal(size=(self.num_user, factor_num)))
        self.Y = sp.csr_matrix(self.rstate.normal(size=(self.num_item, factor_num)))
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
                        default=10, 
                        help='No. of training epochs')
    parser.add_argument('--factors', 
                        type=int, 
                        default=20, 
                        help='latent factor number')
    args = parser.parse_args()

    src = 'ml-100k'
    dataset = WRMFData(src)

    algo = WRMF(dataset.train, lambda_val=args.lambda_val, alpha=args.alpha, 
                iterations=args.epochs, factor_num=args.factors)
    algo.fit()

