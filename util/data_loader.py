'''
@Author: Yu Di
@Date: 2019-09-29 11:10:53
@LastEditors: Yudi
@LastEditTime: 2019-09-30 10:56:44
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: data utils
'''
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data

def load_rate(src='ml-100k'):
    df = pd.read_csv(f'./data/{src}/u.data', sep='\t', header=None, 
                     names=['user', 'item', 'rating', 'timestamp'], engine='python')

    return df

def _split_loo(ratings):
    ratings['rank_latest'] = ratings.groupby(['user'])['timestamp'].rank(method='first', 
                                                                         ascending=False)
    train = ratings[ratings['rank_latest'] > 1]
    test = ratings[ratings['rank_latest'] == 1]
    assert train['user'].nunique() == test['user'].nunique()
    return train[['user', 'item', 'rating', 'timestamp']], test[['user', 'item', 'rating', 'timestamp']]

def _negative_sampling(ratings):
    item_pool = set(ratings.item.unique())

    interact_status = ratings.groupby('user')['item'].apply(set).reset_index()
    interact_status.rename(columns={'item': 'interacted_items'}, inplace=True)
    interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
    interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
    
    return interact_status[['user', 'negative_samples']]
    

def load_mat(src='ml-100k', test_num=100):
    train_data = pd.read_csv(f'./data/{src}/{src}.train.rating', sep='\t', header=None, 
                             names=['user', 'item'], usecols=[0, 1], 
                             dtype={0: np.int32, 1: np.int32}, engine='python')
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    
    train_data = train_data.values.tolist()
    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    
    test_data = []
    with open(f'./data/{src}/{src}.test.negative', 'r') as fd:
        line = fd.readline()
        while line is not None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat

class BPRData(data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        ''' 
            Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
		'''
        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for _ in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if self.is_training else len(self.features)
    
    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if self.is_training else features[idx][1]
        return user, item_i, item_j

if __name__ == '__main__':
    # load negative sampling dataset, take ml-100k as an example
    df = load_rate('ml-100k')
    df.sort_values(by=['user', 'item', 'timestamp'], inplace=True)

    df['user'] = pd.Categorical(df.user).codes
    df['item'] = pd.Categorical(df.item).codes

    negatives = _negative_sampling(df)
    train, test = _split_loo(df)

    file_obj = open('./data/ml-100k/ml-100k.train.rating', 'w')
    for _, row in train.iterrows():
        ln = '\t'.join(map(str, row.values)) + '\n'
        file_obj.write(ln)
    file_obj.close()

    file_obj = open('./data/ml-100k/ml-100k.test.rating', 'w')
    for _, row in test.iterrows():
        ln = '\t'.join(map(str, row.values)) + '\n'
        file_obj.write(ln)
    file_obj.close()

    negs = test.merge(negatives, on=['user'], how='left')
    negs['user'] = negs.apply(lambda x: f'({x["user"]},{x["item"]})', axis=1)
    negs.drop(['item', 'rating', 'timestamp'], axis=1, inplace=True)

    file_obj = open('./data/ml-100k/ml-100k.test.negative', 'w')
    for _, row in negs.iterrows():
        ln = row['user'] + '\t' + '\t'.join(map(str, row['negative_samples'])) + '\n'
        file_obj.write(ln)
    file_obj.close()
