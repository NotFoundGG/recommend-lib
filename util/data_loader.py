'''
@Author: Yu Di
@Date: 2019-09-29 11:10:53
@LastEditors: Yudi
@LastEditTime: 2019-11-14 22:10:13
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: data utils
'''
import os
import gc
import csv
import json
import random
from collections import defaultdict
from collections.abc import MutableMapping

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.utils.data as data
from scipy.sparse import csr_matrix, coo_matrix
from scipy.io import mmread

from sklearn.model_selection import train_test_split, KFold

########################################################################################################
def load_rate(src='ml-100k', prepro='origin'):
    if src == 'ml-100k':
        df = pd.read_csv(f'./data/{src}/u.data', sep='\t', header=None, 
                        names=['user', 'item', 'rating', 'timestamp'], engine='python')
    elif src == 'ml-1m':
        df = pd.read_csv(f'./data/{src}/ratings.dat', sep='::', header=None, 
                        names=['user', 'item', 'rating', 'timestamp'], engine='python')
        # only consider rating >=4 for data density
        df = df.query('rating >= 4').reset_index(drop=True).copy()
    elif src == 'ml-10m':
        df = pd.read_csv(f'./data/{src}/ratings.dat', sep='::', header=None, 
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')
        df = df.query('rating >= 4').reset_index(drop=True).copy()
    elif src == 'ml-20m':
        df = pd.read_csv(f'./data/{src}/ratings.csv')
        df.rename(columns={'userId':'user', 'movieId':'item'}, inplace=True)
        df = df.query('rating >= 4').reset_index(drop=True)
    elif src == 'netflix':
        pass
    elif src == 'lastfm':
        pass
    elif src == 'bx':
        df = pd.read_csv(f'./data/{src}/BX-Book-Ratings.csv', delimiter=";", encoding="latin1")
        df.rename(columns={'User-ID': 'user', 'ISBN': 'item', 'Book-Rating': 'rating'}, inplace=True)
    elif src == 'pinterest':
        pass
    elif src == 'amazon-cloth':
        df = pd.read_csv(f'./data/{src}/ratings_Clothing_Shoes_and_Jewelry.csv', 
                         names=['user', 'item', 'rating', 'timestamp'])
    elif src == 'amazon-electronic':
        df = pd.read_csv(f'./data/{src}/ratings_Electronics.csv', 
                         names=['user', 'item', 'rating', 'timestamp'])
    elif src == 'amazon-book':
        pass
    elif src == 'amazon-music':
        df = pd.read_csv(f'./data/{src}/ratings_Digital_Music.csv', 
                         names=['user', 'item', 'rating', 'timestamp'])
    elif src == 'epinions':
        pass
    elif src == 'yelp':
        json_file_path = f'./data/{src}/yelp_academic_dataset_review.json'
        prime = []
        for line in open(json_file_path, 'r', encoding='UTF-8'):
            val = json.loads(line)
            prime.append([val['user_id'], val['business_id'], val['stars'], val['date']])
        df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])
        df['timestamp'] = pd.to_datetime(df.timestamp)
        del prime
        gc.collect()

    elif src == 'citeulike':
        pass
    else:
        raise ValueError('Invalid Dataset Error')

    df.sort_values(['user', 'item', 'timestamp'], inplace=True)

    if prepro == 'origin':
        return df
    elif prepro == '5core':
        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        df = df.query('cnt_item >= 5 and cnt_user >= 5').reset_index(drop=True).copy()
        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

        return df
    elif prepro == '10core':
        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        df = df.query('cnt_item >= 10 and cnt_user >= 10').reset_index(drop=True).copy()
        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()
        
        return df
    else:
        raise ValueError('Invalid dataset preprocess type, origin/5core/10core expected')

# NeuFM/FM prepare
def load_libfm(src='ml-100k', data_split='fo', by_time=0, val_method='cv', fold_num=5, prepro='origin'):
    df = load_rate(src, prepro)

    if src == 'ml-100k':
        # rating >=4 interaction =1
        df['rating'] = df.rating.agg(lambda x: 1 if x >= 4 else -1).astype(float)

        df['user'] = pd.Categorical(df.user).codes
        df['item'] = pd.Categorical(df.item).codes
        user_tag_info = df[['user']].copy()
        item_tag_info = df[['item']].copy()
        user_tag_info = user_tag_info.drop_duplicates()
        item_tag_info = item_tag_info.drop_duplicates()

    feat_idx_dict = {} # store the start index of each category
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
            
            split_idx = int(np.ceil(len(df) * 0.8)) # for train test
            train, test = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()
        else:
            train, test = train_test_split(df, test_size=0.2)
    elif data_split == 'loo':
        if by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)
            df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
            train, test = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
            train.drop(['rank_latest'], axis=1, inplace=True)
            test.drop(['rank_latest'], axis=1, inplace=True)
        else:
            test = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
            test_key = test[['user', 'item']].copy()
            train = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()
    else:
        raise ValueError('Invalid data_split value, expect: loo, fo')

    test = test.reset_index(drop=True)
    test.drop(['timestamp'], axis=1, inplace=True)

    train_list, val_list = [], []
    if val_method == 'cv':
        kf = KFold(n_splits=fold_num, shuffle=False, random_state=2019)
        for train_index, val_index in kf.split(train):
            train_set, val_set = train.iloc[train_index, :].copy(), train.iloc[val_index, :].copy()
            del train_set['timestamp'], val_set['timestamp']

            train_list.append(train_set)
            val_list.append(val_set)
    elif val_method == 'loo':
        val_set = train.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        val_key = val_set[['user', 'item']].copy()
        train_set = train.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(val_key)).reset_index().copy()
        del train_set['timestamp'], val_set['timestamp']

        train_list.append(train_set)
        val_list.append(val_set)
    elif val_method == 'tloo':
        train = train.sample(frac=1)
        train = train.sort_values(['timestamp']).reset_index(drop=True)

        train['rank_latest'] = train.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        train_set = train[train['rank_latest'] > 1].copy()
        val_set = train[train['rank_latest'] == 1].copy()
        del train_set['rank_latest'], val_set['rank_latest']
        del train_set['timestamp'], val_set['timestamp']

        train_list.append(train_set)
        val_list.append(val_set)
    elif val_method == 'tfo':
        train = train.sample(frac=1)
        train = train.sort_values(['timestamp']).reset_index(drop=True)

        split_idx = int(np.ceil(len(train) * 0.9))
        train_set, val_set = train.iloc[:split_idx, :].copy(), train_set.iloc[split_idx:, :].copy()
        del train_set['timestamp'], val_set['timestamp']

        train_list.append(train_set)
        val_list.append(val_set)
    else:
        raise ValueError('Invalid val_method value, expect: cv, loo, tloo, tfo')

    test_user_set, test_item_set = test['user'].unique().tolist(), test['item'].unique().tolist()
    ui = test[['user', 'item']].copy()
    u_is = defaultdict(list)
    for u in test_user_set:
        u_is[u] = ui.loc[ui.user==u, 'item'].values.tolist()

    gc.collect()
    for fold in range(len(train_list)):
        file_obj = open(f'./data/{src}/{src}.train.libfm.{fold}', 'w')
        for idx, row in train_list[fold].iterrows():
            l = ''
            for col in train_list[fold].columns:
                if col != 'rating':
                    l += ' '
                    l = l + str(int(feat_idx_dict[col] + row[col])) + ':1'
            l = str(row['rating']) + l + '\n'
            file_obj.write(l)
        file_obj.close()

        file_obj = open(f'./data/{src}/{src}.valid.libfm.{fold}', 'w')
        for idx, row in val_list[fold].iterrows():
            l = ''
            for col in val_list[fold].columns:
                if col != 'rating':
                    l += ' '
                    l = l + str(int(feat_idx_dict[col] + row[col])) + ':1'
            l = str(row['rating']) + l + '\n'
            file_obj.write(l)
        file_obj.close()

    file_obj = open(f'./data/{src}/{src}.test.libfm', 'w')
    for idx, row in test.iterrows():
        l = ''
        for col in test.columns:
            if col != 'rating':
                l += ' '
                l = l + str(int(feat_idx_dict[col] + row[col])) + ':1'
        l = str(row['rating']) + l + '\n'
        file_obj.write(l)
    file_obj.close()

    return feat_idx_dict, user_tag_info, item_tag_info, test_user_set, test_item_set, u_is

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
    

def load_mat(src='ml-100k', test_num=100, data_split='loo', by_time=1, val_method='cv', fold_num=5, prepro='origin'):
    df = load_rate(src, prepro)
    # df.sort_values(by=['user', 'item', 'timestamp'], inplace=True)
    df['user'] = pd.Categorical(df.user).codes
    df['item'] = pd.Categorical(df.item).codes

    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1

    if data_split == 'loo':
        negatives = _negative_sampling(df)
        train, test = _split_loo(df, by_time)

        negs = test.merge(negatives, on=['user'], how='left')
        negs['user'] = negs.apply(lambda x: f'({x["user"]},{x["item"]})', axis=1)
        negs.drop(['item', 'rating', 'timestamp'], axis=1, inplace=True)
        
        test_data = []
        ur = defaultdict(set)

        for _, row in negs.iterrows():
            u = eval(row['user'])[0]
            test_data.append([u, eval(row['user'])[1]])
            ur[u].add(int(row['user'][1]))
            for i in row['negative_samples']:
                test_data.append([u, int(i)])

    elif data_split == 'fo':
        negatives = _negative_sampling(df)
        train, test = _split_fo(df, by_time)

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

    train_data_list, val_data_list = [], []
    if val_method == 'cv':
        train_data = train[['user', 'item']].reset_index(drop=True)
        train_data = train_data.values
        kf = KFold(n_splits=fold_num, shuffle=False, random_state=2019)
        for train_index, val_index in kf.split(train_data):
            train_data_list.append(train_data[train_index].tolist())
            val_data_list.append(train_data[val_index].tolist())
    elif val_method == 'tloo':
        train_data = train[['user', 'item', 'timestamp']].reset_index(drop=True)
        train_data = train_data.sample(frac=1)
        train_data = train_data.sort_values(['timestamp']).reset_index(drop=True)

        train_data['rank_latest'] = train_data.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        sub_train = train_data[train_data['rank_latest'] > 1].copy()
        sub_val = train_data[train_data['rank_latest'] == 1].copy()
        sub_val.drop(['rank_latest', 'timestamp'], axis=1, inplace=True)
        sub_train.drop(['rank_latest', 'timestamp'], axis=1, inplace=True)

        train_data_list.append(sub_train)
        val_data_list.append(sub_val)
    elif val_method == 'loo':
        sub_val = train.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        sub_val = sub_val[['user', 'item']].copy()
        sub_train = train.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(sub_val)).reset_index().copy()
        sub_train = sub_train[['user', 'item']].copy()

        train_data_list.append(sub_train)
        val_data_list.append(sub_val)
    elif val_method == 'tfo':
        train_data = train[['user', 'item', 'timestamp']].reset_index(drop=True)
        train_data = train_data.sample(frac=1)
        train_data = train_data.sort_values(['timestamp']).reset_index(drop=True)

        split_idx = int(np.ceil(len(train_data) * 0.9))
        train_data.drop(['timestamp'], axis=1, inplace=True)
        sub_train, sub_val = train_data.iloc[:split_idx, :].copy(), train_data.iloc[split_idx:, :].copy()

        train_data_list.append(sub_train.values.tolist())
        val_data_list.append(sub_val.values.tolist())
    else:
        raise ValueError('Invalid val_method value, expect: cv, loo, tloo, tfo')
    train_mat_list = []
    for fold in range(len(train_data_list)):
        # load ratings as a dok matrix
        train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for x in train_data_list[fold]:
            train_mat[x[0], x[1]] = 1.0

        train_mat_list.append(train_mat)
    print('Finish build train and test set......')
    
    return train_data_list, test_data, user_num, item_num, train_mat_list, ur, val_data_list

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
    features = read_features(f'./data/{src}/{src}.train.libfm.0', features)
    features = read_features(f'./data/{src}/{src}.valid.libfm.0', features)
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

# SLIM data loader 
class SlimData(object):
    def __init__(self, src='ml-100k', data_split='fo', by_time=0, val_method='cv', fold_num=5, prepro='origin'):
        print('Start read raw data')        
        self.df = load_rate(src, prepro)
        self.df['user'] = pd.Categorical(self.df['user']).codes
        self.df['item'] = pd.Categorical(self.df['item']).codes
        self.num_user = self.df.user.nunique()
        self.num_item = self.df.item.nunique()
        train_df, test_df = self.__split_data(data_split, by_time)
        
        train_list, val_list = self.__get_validation(train_df, val_method, fold_num)

        self.train, self.val = [], []
        for i in range(len(train_list)):
            sub_train, sub_val = [], []
            for _, row in train_list[i].iterrows():
                sub_train.append([row['user'], row['item']])
            for _, row in val_list[i].iterrows():
                sub_val.append([row['user'], row['item']])
            self.train.append(sub_train)
            self.val.append(sub_val)

        self.test = []
        for _, row in test_df.iterrows():
            self.test.append([row['user'], row['item']])
        print(f'{len(self.df)} data records, user num: {self.num_user}, item num: {self.num_item}')
        print(f'Use {val_method} validation method......')
        for i in range(len(self.train)):
            print(f'train set [{i + 1}]: {len(self.train[i])} val set [{i + 1}]: {len(self.val[i])}')
        print(f'test set: {len(self.test)}')
        del self.df    

    def __get_validation(self, train_df, val_method, fold_num):
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
            kf = KFold(n_splits=fold_num, shuffle=False, random_state=2019)
            for train_index, val_index in kf.split(train_df):
                train_list.append(train_df.iloc[train_index, :])
                val_list.append(train_df.iloc[val_index, :])
        else:
            raise ValueError('Invalid val_method value, expect: cv, loo, tloo, tfo')
        
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
    def __init__(self, src='ml-100k', data_split='fo', by_time=0, val_method='cv', fold_num=5, prepro='origin'):
        self.df = load_rate(src, prepro)
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

        self._split_train(val_method, fold_num)
    
    def _split_train(self, val_method, fold_num):
        self.train_list, self.val_users_list = [], []
        val_set = self.train.copy()
        val_set[val_set != 0] = 1
        self.val = val_set

        if val_method == 'cv':
            kf = KFold(n_splits=fold_num, shuffle=False, random_state=2019)
            for _, val_index in kf.split(self.val_df):
                sub_training_set = self.train.copy()
                tmp = self.val_df.iloc[val_index, :]
                user_index = [u for u in tmp.user]
                item_index = [i for i in tmp.item]
                sub_training_set[user_index, item_index] = 0
                sub_training_set.eliminate_zeros()

                self.train_list.append(sub_training_set)
                self.val_users_list.append(list(set(user_index)))
        elif val_method == 'tloo':
            self.val_df = self.val_df.sample(frac=1)
            self.val_df = self.val_df.sort_values(['timestamp']).reset_index(drop=True)

            self.val_df['rank_latest'] = self.val_df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
            tmp = self.val_df[self.val_df['rank_latest'] == 1].copy()
            del self.val_df['rank_latest']

            user_index = [u for u in tmp.user]
            item_index = [i for i in tmp.item]
            sub_training_set = self.train.copy()
            sub_training_set[user_index, item_index] = 0
            sub_training_set.eliminate_zeros()

            self.train_list.append(sub_training_set)
            self.val_users_list.append(list(set(user_index)))
        elif val_method == 'loo':
            tmp = self.val_df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
            user_index = [u for u in tmp.user]
            item_index = [i for i in tmp.item]

            sub_training_set = self.train.copy()
            sub_training_set[user_index, item_index] = 0
            sub_training_set.eliminate_zeros()

            self.train_list.append(sub_training_set)
            self.val_users_list.append(list(set(user_index)))
        elif val_method == 'tfo':
            self.val_df = self.val_df.sample(frac=1)
            self.val_df = self.val_df.sort_values(['timestamp']).reset_index(drop=True)

            split_idx = int(np.ceil(len(self.val_df) * 0.9))
            tmp = self.val_df.iloc[split_idx:, :].copy()
            user_index = [u for u in tmp.user]
            item_index = [i for i in tmp.item]
            sub_training_set = self.train.copy()
            sub_training_set[user_index, item_index] = 0
            sub_training_set.eliminate_zeros()

            self.train_list.append(sub_training_set)
            self.val_users_list.append(list(set(user_index)))
        else:
            raise ValueError('Invalid val_method value, expect: cv, loo, tloo, tfo')

        del self.train

    def _split_data(self, pct_test=0.2):
        test_set = self.mat.copy()
        test_set[test_set != 0] = 1
        training_set = self.mat.copy()
        if self.data_split == 'fo':
            if self.by_time:
                self.df = self.df.sample(frac=1)
                self.df = self.df.sort_values(['timestamp']).reset_index(drop=True)
                split_idx = int(np.ceil(len(self.df) * (1 - pct_test)))
                samples = self.df.iloc[split_idx:, :].copy()
                user_index = [u for u in samples.user]
                item_index = [i for i in samples.item]

                self.val_df = self.df.iloc[:split_idx, :].reset_index(drop=True)
            else:
                val_df, samples = train_test_split(self.df, test_size=pct_test)
                user_index = [u for u in samples.user]
                item_index = [i for i in samples.item]

                self.val_df = val_df.reset_index(drop=True)
        elif self.data_split == 'loo':
            if self.by_time:
                self.df = self.df.sample(frac=1)
                self.df = self.df.sort_values(['timestamp']).reset_index(drop=True)
                self.df['rank_latest'] = self.df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
                samples = self.df[self.df['rank_latest'] == 1].copy()
                user_index = [u for u in samples.user]
                item_index = [i for i in samples.item]

                self.val_df = self.df[self.df['rank_latest'] > 1].copy()
                del self.val_df['rank_latest']
            else:
                samples = self.df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
                user_index = [u for u in samples.user]
                item_index = [i for i in samples.item]

                val_key = samples[['user', 'item']].copy()
                self.val_df = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(val_key)).reset_index().copy()
        else:
            raise ValueError('Invalid data_split value, expect: loo, fo')
        training_set[user_index, item_index] = 0
        # eliminate stored-zero then save space
        training_set.eliminate_zeros()
        # Output the unique list of user rows that were altered; set() for eliminate repeated user_index
        return training_set, test_set, list(set(user_index))

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