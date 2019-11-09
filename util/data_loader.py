'''
@Author: Yu Di
@Date: 2019-09-29 11:10:53
@LastEditors: Yudi
@LastEditTime: 2019-11-04 14:39:43
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: data utils
'''
import os
import gc
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.utils.data as data
from scipy.sparse import csr_matrix, coo_matrix
from scipy.io import mmread

from sklearn.model_selection import train_test_split, KFold

ML100K_NUMERIC_COLS = ['age']
IGNORE_COLS = ['user', 'item']
TARGET_COLS = ['rating']

# SLIM data loader 
class SlimData(object):
    def __init__(self, src='ml-100k', data_split='fo', by_time=0, val_method='cv'):
        print('Start read raw data')        
        self.df = load_rate(src)
        self.df['user'] = pd.Categorical(self.df['user']).codes
        self.df['item'] = pd.Categorical(self.df['item']).codes
        self.num_user = self.df.user.nunique()
        self.num_item = self.df.item.nunique()
        train_df, test_df = self.__split_data(data_split, by_time)
        
        self.train_list, self.val_list = self.__get_validation(train_df, val_method)

        self.train, self.test = [], []
        for _, row in train_df.iterrows():
            self.train.append([row['user'], row['item']])
        for _, row in test_df.iterrows():
            self.test.append([row['user'], row['item']])
        print(f'{len(self.df)} data records, train set: {len(self.train)}, test set: {len(self.test)}, user num: {self.num_user}, item num: {self.num_item}')
        del self.df

    def __get_validation(self, train_df, val_method):
        train_list, val_list = [], []
        if val_method == 'loo':
            val_set = train_df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
            val_key = val_set[['user', 'item']].copy()
            train_set = train_df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(val_key)).reset_index().copy()

            train_list.append(train_set)
            val_list.append(val_set)
        elif val_method == 'tfo':
            train_df = train_df.sample(frac=1)
            train_df = train_df.sort_values(['timestamp']).reset_index(drop=True)

            split_idx = int(np.ceil(len(train_df) * 0.9))
            train_set, val_set = train_df.iloc[:split_idx, :].copy(), train_df.iloc[split_idx:, :].copy()

            train_list.append(train_set)
            val_list.append(val_set)
        elif val_method == 'tloo':
            train_df = train_df.sample(frac=1)
            train_df = train_df.sort_values(['timestamp']).reset_index(drop=True)

            train_df['rank_latest'] = train_df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
            new_train_set = train_df[train_df['rank_latest'] > 1].copy()
            val_set = train_df[train_df['rank_latest'] == 1].copy()
            del new_train_set['rank_latest'], val_set['rank_latest']

            train_list.append(new_train_set)
            val_list.append(val_set)
        elif val_method == 'cv':
            kf = KFold(n_splits=5, shuffle=False, random_state=2019)
            for train_index, val_index in kf.split(train_df):
                train_list.append(train_df.iloc[train_index, :])
                val_list.append(train_df.iloc[val_index, :])
        else:
            raise ValueError('Invalid data_split value, expect: loo, fo')
        
        return train_list, val_list

    def __split_data(self, data_split, by_time):
        '''without time stemp'''
        if data_split == 'fo':
            if by_time:
                self.df = self.df.sample(frac=1)
                self.df = self.df.sort_values(['timestamp']).reset_index(drop=True)
                split_idx = int(np.ceil(len(self.df) * 0.8))
                train, test = self.df.iloc[:split_idx, :].copy(), self.df.iloc[split_idx:, :].copy()
                return train, test
            else:
                train, test = train_test_split(self.df, test_size=.2)
                return train, test
        elif data_split == 'loo':
            if by_time:
                self.df = self.df.sample(frac=1)
                self.df = self.df.sort_values(['timestamp']).reset_index(drop=True)
                self.df['rank_latest'] = self.df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
                train, test = self.df[self.df['rank_latest'] > 1].copy(), self.df[self.df['rank_latest'] == 1].copy()
                del train['rank_latest'], test['rank_latest']
                return train, test
            else:
                test = self.df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
                test_key = test[['user', 'item']].copy()
                train = self.df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()
                return train, test
        else:
            raise ValueError('Invalid data_split value, expect: loo, fo')
            
########################################################################################################
class WRMFData(object):
    def __init__(self, src='ml-100k', data_split='fo', by_time=0):
        self.df = load_rate(src)
        self.data_split = data_split
        self.by_time = by_time

        ratings = list(self.df.rating)
        rows = pd.Categorical(self.df.user).codes
        cols = pd.Categorical(self.df.item).codes
        self.df['user'] = rows
        self.df['item'] = cols
        self.user_num, self.item_num = self.df.user.nunique(), self.df.item.nunique()
        
        self.mat = sp.csr_matrix((ratings, (rows, cols)), shape=(self.user_num, self.item_num))
        self.train, self.test, self.test_users = self._split_data()

    def _split_data(self, pct_test=0.2):
        test_set = self.mat.copy()
        test_set[test_set != 0] = 1
        training_set = self.mat.copy()
        if self.data_split == 'fo':
            if self.by_time:
                self.df = self.df.sample(frac=1)
                self.df = self.df.sort_values(['timestamp']).reset_index(drop=True)
                split_idx = int(np.ceil(len(self.df) * 0.8))
                samples = self.df.iloc[split_idx:, :].copy()
                user_index = [u for u in samples.user]
                item_index = [i for i in samples.item]
            else:
                nonzero_index = training_set.nonzero()
                nonzero_pairs = list(zip(nonzero_index[0], nonzero_index[1]))
                random.seed(0)
                # Round the number of samples needed to the nearest integer
                num_samples = int(np.ceil(pct_test * len(nonzero_pairs)))
                # remove num_samples values from nonzero_pairs
                samples = random.sample(nonzero_pairs, num_samples)
                user_index = [index[0] for index in samples]
                item_index = [index[1] for index in samples]
        elif self.data_split == 'loo':
            if self.by_time:
                self.df = self.df.sample(frac=1)
                self.df = self.df.sort_values(['timestamp']).reset_index(drop=True)
                self.df['rank_latest'] = self.df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
                samples = self.df[self.df['rank_latest'] == 1].copy()
                user_index = [u for u in samples.user]
                item_index = [i for i in samples.item]
            else:
                samples = self.df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
                user_index = [u for u in samples.user]
                item_index = [i for i in samples.item]
        else:
            raise ValueError('Invalid data_split value, expect: loo, fo')
        training_set[user_index, item_index] = 0
        # eliminate stored-zero then save space
        training_set.eliminate_zeros()
        # Output the unique list of user rows that were altered; set() for eliminate repeated user_index
        return training_set, test_set, list(set(user_index))

########################################################################################################
def load_rate(src='ml-100k'):
    if src == 'ml-100k':
        df = pd.read_csv(f'./data/{src}/u.data', sep='\t', header=None, 
                        names=['user', 'item', 'rating', 'timestamp'], engine='python')
        df.sort_values(['user', 'item', 'timestamp'], inplace=True)
    elif src == 'netflix':
        # df = pd.DataFrame()
        # for i in range(1, 5):
        #     tmp = pd.read_csv(f'./data/netflix/combined_data_{i}.txt', header=None, names = ['user', 'rating'], usecols=[0, 1])
        #     tmp['rating'] = tmp.rating.astype(float)
        #     df = pd.concat([df, tmp], ignore_index=True)
        #     del tmp
        #     gc.collect()
        # df_nan = pd.DataFrame(pd.isna(df.rating))
        # df_nan = df_nan[df_nan['rating']==True]
        # df_nan = df_nan.reset_index()
        # movie_np = []
        # movie_id = 1
        # for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        #     temp = np.full((1, i-j-1), movie_id)
        #     movie_np = np.append(movie_np, temp)
        #     movie_id += 1
        # last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
        # movie_np = np.append(movie_np, last_record)

        # df = df[pd.notnull(df['rating'])]
        # df['item'] = movie_np.astype(int)
        # df['user'] = df['user'].astype(int)
        pass


    return df

# NeuFM/FM prepare
def load_libfm(src='ml-100k', data_split='fo', by_time=0):
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
                    'zip_code', 'release_date'], axis=1, inplace=True)

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
        if col not in ['rating', 'timestamp']:
            feat_idx_dict[col] = idx
            idx = idx + df[col].max() + 1
    print('Finish build category index dictionary......')

    if data_split == 'fo':
        if by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)
            del df['timestamp']
            
            split_idx = int(np.ceil(len(df) * 0.8)) # for train test
            train, test = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()
            split_idx = int(np.ceil(len(train) * 0.9)) # for train valid
            train, valid = train.iloc[:split_idx, :].copy(), train.iloc[split_idx:, :].copy()
        else:
            del df['timestamp']
            train, test = train_test_split(df, test_size=0.2)
            train, valid = train_test_split(train, test_size=0.1)
    elif data_split == 'loo':
        if by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)
            df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
            train, test = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
            train.drop(['rank_latest', 'timestamp'], axis=1, inplace=True)
            test.drop(['rank_latest', 'timestamp'], axis=1, inplace=True)
            split_idx = int(np.ceil(len(train) * 0.9)) # for train valid
            train = train.reset_index(drop=True)
            train, valid = train.iloc[:split_idx, :].copy(), train.iloc[split_idx:, :].copy()
        else:
            del df['timestamp']
            test = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
            test_key = test[['user', 'item']].copy()
            train = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()
            train, valid = train_test_split(train, test_size=0.1)
    else:
        raise ValueError('Invalid data_split value, expect: loo, fo')
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
def _split_loo(ratings, by_time=1):
    if by_time:
        ratings['rank_latest'] = ratings.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        train = ratings[ratings['rank_latest'] > 1].copy()
        test = ratings[ratings['rank_latest'] == 1].copy()
    else:
        ratings = ratings.sample(frac=1)
        test = ratings.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        test_key = test[['user', 'item']].copy()
        train = ratings.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()
    assert train['user'].nunique() == test['user'].nunique()
    return train[['user', 'item', 'rating', 'timestamp']], test[['user', 'item', 'rating', 'timestamp']]

def _split_fo(ratings, by_time=0):
    if by_time:
        ratings = ratings.sample(frac=1)
        ratings = ratings.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(ratings) * 0.8))
        train, test = ratings.iloc[:split_idx, :].copy(), ratings.iloc[split_idx:, :].copy()
    else:
        train, test = train_test_split(ratings, test_size=.2)
    return train[['user', 'item', 'rating', 'timestamp']], test[['user', 'item', 'rating', 'timestamp']]

def _negative_sampling(ratings):
    item_pool = set(ratings.item.unique())

    interact_status = ratings.groupby('user')['item'].apply(set).reset_index()
    interact_status.rename(columns={'item': 'interacted_items'}, inplace=True)
    interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
    interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
    
    return interact_status[['user', 'negative_samples']]
    

def load_mat(src='ml-100k', test_num=100, data_split='loo', by_time=1):
    df = load_rate(src)
    # df.sort_values(by=['user', 'item', 'timestamp'], inplace=True)
    df['user'] = pd.Categorical(df.user).codes
    df['item'] = pd.Categorical(df.item).codes

    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1

    if data_split == 'loo':
        negatives = _negative_sampling(df)
        train, test = _split_loo(df, by_time)

        file_obj = open(f'./data/{src}/{src}.train.rating', 'w')
        for _, row in train.iterrows():
            ln = '\t'.join(map(str, row.values)) + '\n'
            file_obj.write(ln)
        file_obj.close()

        file_obj = open(f'./data/{src}/{src}.test.rating', 'w')
        for _, row in test.iterrows():
            ln = '\t'.join(map(str, row.values)) + '\n'
            file_obj.write(ln)
        file_obj.close()

        negs = test.merge(negatives, on=['user'], how='left')
        negs['user'] = negs.apply(lambda x: f'({x["user"]},{x["item"]})', axis=1)
        negs.drop(['item', 'rating', 'timestamp'], axis=1, inplace=True)

        file_obj = open(f'./data/{src}/{src}.test.negative', 'w')
        for _, row in negs.iterrows():
            ln = row['user'] + '\t' + '\t'.join(map(str, row['negative_samples'])) + '\n'
            file_obj.write(ln)
        file_obj.close()
        
        test_data = []
        ur = defaultdict(set)
        with open(f'./data/{src}/{src}.test.negative', 'r') as fd:
            line = fd.readline()
            while line is not None and line != '':
                arr = line.split('\t')
                u = eval(arr[0])[0]
                test_data.append([u, eval(arr[0])[1]])
                ur[u].add(int(eval(arr[0])[1]))
                for i in arr[1:]:
                    test_data.append([u, int(i)])
                line = fd.readline()

    elif data_split == 'fo':
        negatives = _negative_sampling(df)
        train, test = _split_fo(df, by_time)

        file_obj = open(f'./data/{src}/{src}.train.rating', 'w')
        for _, row in train.iterrows():
            ln = '\t'.join(map(str, row.values)) + '\n'
            file_obj.write(ln)
        file_obj.close()

        file_obj = open(f'./data/{src}/{src}.test.rating', 'w')
        for _, row in test.iterrows():
            ln = '\t'.join(map(str, row.values)) + '\n'
            file_obj.write(ln)
        file_obj.close()

        test_data = []
        ur = defaultdict(set) # ground_truth
        max_i_num = 100
        for u in test.user.unique():
            pre_cands = negatives.query(f'user=={u}')['negative_samples'].values[0]  # 99 pre-candidates
            test_u_is = test.query(f'user=={u}')['item'].values.tolist()
            ur[u] = set(test_u_is)
            if len(test_u_is) < max_i_num:
                cands_num = max_i_num - len(test_u_is)
                candidates = random.sample(pre_cands, cands_num)
                test_u_is = list(set(candidates) | set(test_u_is))
            else:
                test_u_is = random.sample(test_u_is, max_i_num)
            for i in test_u_is:
                test_data.append([u, i])
    else:
        raise ValueError('Invalid data_split value, expect: loo, fo')

    train_data = pd.read_csv(f'./data/{src}/{src}.train.rating', sep='\t', header=None, 
                            names=['user', 'item'], usecols=[0, 1], 
                            dtype={0: np.int32, 1: np.int32}, engine='python')

    train_data = train_data.values.tolist()
    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    print('Finish build train and test set......')
    
    return train_data, test_data, user_num, item_num, train_mat, ur

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