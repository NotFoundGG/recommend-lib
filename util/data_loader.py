'''
@Author: Yu Di
@Date: 2019-09-29 11:10:53
@LastEditors: Yudi
@LastEditTime: 2019-10-31 15:06:41
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: data utils
'''
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.utils.data as data
from scipy.sparse import csr_matrix, coo_matrix
from scipy.io import mmread

from sklearn.model_selection import train_test_split

ML100K_NUMERIC_COLS = ['age']
IGNORE_COLS = ['user', 'item']
TARGET_COLS = ['rating']

# SLIM data loader 
class SlimData(object):
    def __init__(self, src='ml-100k'):
        path = None
        seperator = None
        if src == 'ml-100k':
            path = f'./data/{src}/u.data'
            seperator = '\t'
        elif src == 'ml-1m':
            pass

        print('Start read raw data')
        self.data = []
        for line in open(path, 'r'):
            data_line = line.split(seperator)
            userID = int(data_line[0])
            movieID = int(data_line[1])
            self.data.append([userID, movieID])
        
        def compress(data, col):
            e_rows = dict()
            for i in range(len(data)):
                e = data[i][col]
                if e not in e_rows:
                    e_rows[e] = []
                e_rows[e].append(i)

            for rows, i in zip(e_rows.values(), range(len(e_rows))):
                for row in rows:
                    data[row][col] = i

            return len(e_rows)

        self.num_user = compress(self.data, 0)
        self.num_item = compress(self.data, 1)

        self.train, self.test = self.__split_data()
        print(f'{len(self.data)} data records, train set: {len(self.train)}, \
                 test set: {len(self.test)}, user num: {self.num_user}, item num: {self.num_item}')
    
    def __split_data(self):
        '''without time stemp'''
        train, test = train_test_split(self.data, test_size=.2)
        return train, test
########################################################################################################

def load_rate(src='ml-100k'):
    if src == 'ml-100k':
        df = pd.read_csv(f'./data/{src}/u.data', sep='\t', header=None, 
                        names=['user', 'item', 'rating', 'timestamp'], engine='python')
        df.sort_values(['user', 'item', 'timestamp'], inplace=True)

    return df

# NeuFM/FM prepare
def load_libfm(src='ml-100k'):
    df = load_rate(src)

    if src == 'ml-100k':
        user_info = pd.read_csv(f'./data/{src}/u.user', sep='|', header=None, engine='python', 
                                names=['user', 'age', 'gender', 'occupation', 'zip_code'])
        item_info = pd.read_csv(f'./data/{src}/u.item', sep='|', header=None, engine='python',
                                names=['item', 'movie_title', 'release_date', 'video_release_date', 
                                    'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                                    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                                    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                    'Thriller', 'War', 'Western'])

        df = df.merge(user_info, on='user', how='left').merge(item_info, on='item', how='left')
        df.drop(['IMDb_URL', 'video_release_date', 'movie_title', 
                 'zip_code', 'timestamp', 'release_date'], axis=1, inplace=True)

        # rating >=4 interaction =1
        df['rating'] = df.rating.agg(lambda x: 1 if x >= 4 else -1).astype(float)

        df['user'] = pd.Categorical(df.user).codes
        df['item'] = pd.Categorical(df.item).codes
        df['gender'] = pd.Categorical(df.gender).codes
        df['occupation'] = pd.Categorical(df.occupation).codes
        df = df[[col for col in df.columns if col not in ML100K_NUMERIC_COLS]].copy()

        user_tag_info = df[['user', 'gender', 'occupation']].copy()
        item_tag_info = df[['item', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']].copy()
        user_tag_info = user_tag_info.drop_duplicates()
        item_tag_info = item_tag_info.drop_duplicates()

    feat_idx_dict = {} # 存储各个category特征的起始索引位置
    idx = 0
    for col in df.columns:
        if col != 'rating':
            feat_idx_dict[col] = idx
            idx = idx + df[col].max() + 1

    train, test = train_test_split(df, test_size=0.1)
    train, valid = train_test_split(train, test_size=0.1)

    test_user_set, test_item_set = test['user'].unique().tolist(), test['item'].unique().tolist()
    ui = test[['user', 'item']].copy()
    u_is = defaultdict(list)
    for u in test_user_set:
        u_is[u] = ui.loc[ui.user==u, 'item'].values.tolist()

    file_obj = open(f'./data/{src}/{src}.train.libfm', 'w')
    for idx, row in train.iterrows():
        l = ''
        for col in df.columns:
            if col != 'rating':
                l += ' '
                l = l + str(int(feat_idx_dict[col] + row[col])) + ':1'
        l = str(row['rating']) + l + '\n'
        file_obj.write(l)
    file_obj.close()

    file_obj = open(f'./data/{src}/{src}.test.libfm', 'w')
    for idx, row in test.iterrows():
        l = ''
        for col in df.columns:
            if col != 'rating':
                l += ' '
                l = l + str(int(feat_idx_dict[col] + row[col])) + ':1'
        l = str(row['rating']) + l + '\n'
        file_obj.write(l)
    file_obj.close()

    file_obj = open(f'./data/{src}/{src}.valid.libfm', 'w')
    for idx, row in valid.iterrows():
        l = ''
        for col in df.columns:
            if col != 'rating':
                l += ' '
                l = l + str(int(feat_idx_dict[col] + row[col])) + ':1'
        l = str(row['rating']) + l + '\n'
        file_obj.write(l)
    file_obj.close()

    return feat_idx_dict, user_tag_info, item_tag_info, test_user_set, test_item_set, u_is
        

def read_features(file, features):
    ''' Read features from the given file. '''
    i = len(features)
    with open(file, 'r') as fd:
        line = fd.readline()
        while line:
            items = line.strip().split()
            for item in items[1:]:
                item = item.split(':')[0]
                if item not in features:
                    features[item] = i
                    i += 1
            line = fd.readline()
    return features

def map_features(src='ml-100k'):
    features = {}
    features = read_features(f'./data/{src}/{src}.train.libfm', features)
    features = read_features(f'./data/{src}/{src}.valid.libfm', features)
    features = read_features(f'./data/{src}/{src}.test.libfm', features)
    print(f'number of features: {len(features)}')

    return features, len(features)

class FMData(data.Dataset):
    ''' Construct the FM pytorch dataset. '''
    def __init__(self, file, feature_map, loss_type='square_loss'):
        super(FMData, self).__init__()
        self.label = []
        self.features = []
        self.feature_values = []
        assert loss_type in ['square_loss', 'log_loss']

        with open(file, 'r') as fd:
            line = fd.readline()

            while line:
                items = line.strip().split()
                # convert features
                raw = [item.split(':')[0] for item in items[1:]]
                self.features.append(np.array([feature_map[item] for item in raw], dtype=np.int64))
                self.feature_values.append(np.array([item.split(':')[1] for item in items[1:]], 
                                           dtype=np.float32))
                # convert labels
                if loss_type == 'square_loss':
                    self.label.append(np.float32(items[0]))
                else: # log_loss
                    label = 1 if float(items[0]) > 0 else 0
                    self.label.append(label)

                line = fd.readline()
        assert all(len(item) == len(self.features[0]) for item in self.features), 'features are of different length'

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        features = self.features[idx]
        feature_values = self.feature_values[idx]
        return features, feature_values, label

###############

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

class NCFData(data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        '''
        Note that the labels are only useful when training, we thus 
		add them in the ng_sample() function.
        '''
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for _ in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])
        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label

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
    # load negative sampling dataset for NCF BPR, take ml-100k as an example
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

    # load dataset with features for DeepFM, take ml-100k as an example
    # train_df, test_df, _ = load_features('ml-100k')