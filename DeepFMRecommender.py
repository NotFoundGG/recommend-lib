'''
@Author: Yu Di
@Date: 2019-09-30 15:27:46
@LastEditors: Yudi
@LastEditTime: 2019-10-01 19:03:07
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import os
import argparse
from time import time
import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_fixlen_feature_names

from util.data_loader import load_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', 
                        type=int, 
                        default=2, 
                        help='learning rate')
    parser.add_argument('--epoch', 
                        type=int, 
                        default=20, 
                        help='epoch for training')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256, 
                        help='batch size for training')
    
    args = parser.parse_args()
    
    src = 'ml-100k'

    data, sparse_features, dense_features = load_features(src)
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)
    target = ['rating']

    # Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) 
                                for feat in sparse_features] + [DenseFeat(feat, 1,) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

    # generate input data for model
    train, test = train_test_split(data, test_size=.2)
    train_model_input = [train[name] for name in fixlen_feature_names]
    test_model_input = [test[name] for name in fixlen_feature_names]

    # define model, train, predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.compile('adam', loss='mse', metrics=['mse'])

    history = model.fit(train_model_input, train[target].values, batch_size=args.batch_size, 
                        epochs=args.epoch, verbose=args.verbose, validation_split=.1)

    pred_ans = model.predict(test_model_input, batch_size=256)
    print('test MSE', round(mean_squared_error(test[target].values, pred_ans), 4))
