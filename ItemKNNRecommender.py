'''
@Author: Yu Di
@Date: 2019-09-29 10:54:50
@LastEditors: Yudi
@LastEditTime: 2019-09-29 16:04:18
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Item-KNN recommender
'''
import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

from data_loader import load_rate
from metrics import ndcg_at_k, mean_average_precision

if __name__ == '__main__':
    k = 10

    df = load_rate('ml100k')

    reader = Reader()
    data = Dataset.load_from_df(df=df[['user', 'item', 'rating']], reader=reader, rating_scale=(1, 5))
    train_set, test_set = train_test_split(data, test_size=.2)

    # params for item KNN
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(train_set)

    test_set = pd.DataFrame(test_set, columns=['user', 'item', 'rating']) 
    
    # get top-N list for test users
    preds = {}
    item_list = test_set.item.unique()
    for u in test_set.user.unique():
        pred_rates = [algo.predict(u, i)[0] for i in item_list]
        rec_idx = np.argsort(pred_rates)[::-1][:k]
        top_n = item_list[rec_idx]
        preds[u] = list(top_n)
    # get actual interaction info. of test users
    ur = defaultdict(list)
    for u in test_set.user.unique():
        ur[u] = test_set.loc[test_set.user==u, 'item'].values.tolist()
    for u in preds.keys():
        preds[u] = [1 if e in ur[u] else 0 for e in preds[u]]

    # calculate metrics
    map_k = mean_average_precision(list(preds.values()))
    print(f'MAP@{k}: {map_k}')

    ndcg_k = np.mean([ndcg_at_k(r, k) for r in preds.values()])
    print(f'NDCG@{k}: {ndcg_k}')

