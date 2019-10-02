'''
@Author: Yu Di
@Date: 2019-09-30 15:27:46
@LastEditors: Yudi
@LastEditTime: 2019-10-02 18:12:43
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: DeepFM recommender
'''
import os
import argparse
from time import time
import numpy as np
import pandas as pd
from itertools import chain
from collections import defaultdict

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_fixlen_feature_names

from util.data_loader import load_features, _negative_sampling
from util.metrics import mean_average_precision, hr_at_k, ndcg_at_k

def construct_u_pred_feature(user, item_set, user_info, item_info):
    df = pd.DataFrame({'user': [user for _ in item_set], 'item': item_set})
    df = df.merge(user_info, on='user', how='left').merge(item_info, on='item', how='left')

    return df

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
    parser.add_argument('--embed_size', 
                        type=int, 
                        default=8, 
                        help='embedding size')
    parser.add_argument('--reg_ln', 
                        type=float, 
                        default=0.0001, 
                        help='regularization of linear part')
    parser.add_argument('--reg_embed', 
                        type=float, 
                        default=0.0001, 
                        help='regularization of embedding part')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0, 
                        help='drop out rate')

    parser.add_argument('--hid_units', 
                        type=tuple, 
                        nargs='+', 
                        default=(128, 128),
                        help='hidden units architecture for DNN')

    parser.add_argument('--top_k', 
                        type=int, 
                        default=10, 
                        help='number of recommend items')
    
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
    model = DeepFM(linear_feature_columns, dnn_feature_columns, embedding_size=args.embed_size, 
                    dnn_dropout=args.dropout, l2_reg_embedding=args.reg_ln, l2_reg_linear=args.reg_embed, 
                    dnn_hidden_units=args.hid_units, task='regression')
    model.compile('adam', loss='mse', metrics=['mse'])

    history = model.fit(train_model_input, train[target].values, batch_size=args.batch_size, 
                        epochs=args.epoch, verbose=args.verbose, validation_split=.1)

    pred_ans = model.predict(test_model_input, batch_size=256)
    print('test MSE', round(mean_squared_error(test[target].values, pred_ans), 4))


    # KPI calculation for test set
    k = args.top_k

    user_set = test.user.unique().tolist()
    item_set = test.item.unique().tolist()

    ui = defaultdict(list)
    for u in user_set:
        i_list = test.loc[test.user==u, 'item'].values.tolist()
        ui[u] = i_list

    # test item negative sampling
    negatives = _negative_sampling(data)

    user_info = pd.read_csv(f'./data/{src}/u.user', sep='|', header=None, engine='python', 
                            names=['user', 'age', 'gender', 'occupation', 'zip_code'])
    item_info = pd.read_csv(f'./data/{src}/u.item', sep='|', header=None, engine='python',
                            names=['item', 'movie_title', 'release_date', 'video_release_date', 
                                    'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                                    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                                    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                    'Thriller', 'War', 'Western'])

    print('Start generate rank list......')
    start_time = time()
    rs, ndcg_list = [], []
    for u in user_set:
        items = negatives.query(f'user == {u}')['negative_samples'].values.tolist()[0]
        pred_data = construct_u_pred_feature(u, items, user_info, item_info)

        if src == 'ml-100k':
            pred_data.drop(['IMDb_URL', 'video_release_date', 'movie_title', 
                'zip_code', 'release_date'], axis=1, inplace=True)

        for feat in sparse_features:
            lbe = LabelEncoder()
            pred_data[feat] = lbe.fit_transform(pred_data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        pred_data[dense_features] = mms.fit_transform(pred_data[dense_features])

        fixlen_feature_columns = [SparseFeat(feat, pred_data[feat].nunique()) 
                                    for feat in sparse_features] + [DenseFeat(feat, 1,) for feat in dense_features]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

        pred_model_input = [pred_data[name] for name in fixlen_feature_names]

        # predict result
        pred_ans = model.predict(pred_model_input, batch_size=256)
        pred_ans = list(chain(*pred_ans))
        indices = np.argsort(pred_ans)[::-1][:k]
        rec_list = np.array(items)[indices]

        true_i = ui[u]
        r = [1 if ele in true_i else 0 for ele in rec_list]
        
        rs.append(r)
        ndcg_list.append(ndcg_at_k(r, k))

    print(f'finish prediction, Time cost {time() - start_time}......')

    # calculate metrics
    map_k = mean_average_precision(rs)
    print(f'MAP@{k}: {map_k}')

    ndcg_k = np.mean(ndcg_list)
    print(f'NDCG@{k}: {ndcg_k}')

    hr_k = hr_at_k(rs)
    print(f'HR@{k}: {hr_k}')