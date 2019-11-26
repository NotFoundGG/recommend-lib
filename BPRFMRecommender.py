'''
@Author: Yu Di
@Date: 2019-11-22 11:24:47
@LastEditors: Yudi
@LastEditTime: 2019-11-26 13:41:59
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: BPR-FM
'''
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from util.data_loader import load_bprfm, BPRFMData
from util.metrics import hr_at_k, map_at_k, precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k

class BPRFM(nn.Module):
    def __init__(self, num_features, num_factors, batch_norm, drop_prob):
        '''
        num_features: number of features,
        num_factors: number of hidden factors,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        '''
        super(BPRFM, self).__init__()
        self.num_features = num_features
        self.num_factors = num_factors
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))	
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)

    def forward(self, features_i, feature_values_i, features_j, feature_values_j):
        pred_i = self._out(features_i, feature_values_i)
        pred_j = self._out(features_j, feature_values_j)

        return pred_i, pred_j

    def _out(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM).sum(dim=1, keepdim=True)

        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_

        return FM.view(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepro', 
                        type=str, 
                        default='origin', 
                        help='dataset type for experiment, origin, 5core, 10core available')
    parser.add_argument('--topk', 
                        type=int, 
                        default=10, 
                        help='compute metrics@K')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    parser.add_argument('--data_split', 
                        type=str, 
                        default='loo', 
                        help='method for split test,options: loo/fo')
    parser.add_argument('--by_time', 
                        type=int, 
                        default=1, 
                        help='whether split data by time stamp')
    parser.add_argument('--val_method', 
                        type=str, 
                        default='tfo', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    # model setting
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    parser.add_argument('--batch_norm', 
                        default=True, 
                        help='use batch_norm or not')
    parser.add_argument('--dropout',
                        default='[0.5, 0.2]', 
                        help='dropout rate for FM and MLP')
    parser.add_argument('--opt', 
                        type=str, 
                        default='Adagrad', 
                        help='type of optimizer')
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=4, 
                        help='sample negative items for training')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help='training epochs')
    parser.add_argument('--test_num_ng', 
                        type=int, 
                        default=99, 
                        help='sample part of negative items for testing')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=4096, 
                        help='batch size for training')
    parser.add_argument('--hidden_factor', 
                        type=int, 
                        default=64, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.05, 
                        help='learning rate')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    ### prepare dataset ###
    features_map, train_list, val_list, feat_idx_dict, user_tag_info, item_tag_info,  \
    test_user_set, test_item_set, test_ur = load_bprfm(args.dataset, data_split=args.data_split, 
                                                       by_time=args.by_time, val_method=args.val_method, 
                                                       fold_num=args.fold_num, prepro=args.prepro)
    num_features = len(features_map)
    num_item = len(item_tag_info)

    if args.val_method in ['tloo', 'loo', 'tfo']:
        fn = 1
    elif args.val_method == 'cv':
        fn = args.fold_num
    else:
        raise ValueError('Invalid val_method value')

    max_i_num = 100
    item_pool = set(range(num_item))
    fnl_precision, fnl_recall, fnl_map, fnl_ndcg, fnl_hr, fnl_mrr = [], [], [], [], [], []
    for fold in range(fn):
        print(f'Start train Validation [{fold + 1}]......')
        train_ur = defaultdict(set)
        for _, row in train_list[fold].iterrows():
            train_ur[row['user']].add(row['item'] - feat_idx_dict['item'])

        train_dataset = BPRFMData(train_list[fold], feat_idx_dict, features_map, 
                                  num_item, args.num_ng, True)
        # val_dataset = BPRFMData(val_list[fold], feat_idx_dict, features_map, num_item, 0, False)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                       shuffle=True, num_workers=4)
        # val_loader = data.DataLoader(val_dataset, batch_size=args.test_num_ng + 1, 
        #                              shuffle=False, num_workers=0)
        
        model = BPRFM(num_features, args.hidden_factor, args.batch_norm, eval(args.dropout))

        # model.cuda()
        if torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()

        if args.opt == 'Adagrad':
                optimizer = optim.Adagrad(
                model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
        elif args.opt == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.opt == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.opt == 'Momentum':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)
        
        for epoch in range(args.epochs):
            model.train()
            train_loader.dataset.ng_sample()
            start_time = time.time()

            for feat_i, feat_val_i, feat_j, feat_val_j in train_loader:
                if torch.cuda.is_available():
                    feat_i = feat_i.cuda()
                    feat_j = feat_j.cuda()
                    feat_val_i = feat_val_i.cuda()
                    feat_val_j = feat_val_j.cuda()
                else:
                    feat_i = feat_i.cpu()
                    feat_j = feat_j.cpu()
                    feat_val_i = feat_val_i.cpu()
                    feat_val_j = feat_val_j.cpu()
                
                model.zero_grad()
                pred_i, pred_j = model(feat_i, feat_val_i, feat_j, feat_val_j)
                loss = -(pred_i - pred_j).sigmoid().log().sum()
                loss.backward()
                optimizer.step()

            model.eval()
            elapsed_time = time.time() - start_time
            print('The time elapse of epoch {:03d}'.format(epoch + 1) + ' is: ' + 
                  time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        test_u_is = defaultdict(set)
        for k, v in test_ur.items():
            test_u_is[k] = set(v)
        preds = {}
        # construct candidates set
        for u in test_user_set:
            if len(test_u_is[u]) < max_i_num:
                actual_i_list = test_ur[u]
                cands_num = max_i_num - len(actual_i_list)
                sub_item_pool = item_pool - train_ur[u]
                cands = random.sample(sub_item_pool, cands_num)
                test_u_is[u] = test_u_is[u] | set(cands)
            else:
                test_u_is[u] = set(random.sample(test_u_is[u], max_i_num))
        # calculate KPI
        print('Start Calculating KPI')
        for user in tqdm(test_u_is.keys()):
            test_df = pd.DataFrame(columns=['user', 'item'])
            items = list(test_u_is[user])
            pred_rates = []
            for item in items:
                features_i = torch.LongTensor([[user + feat_idx_dict['user'], 
                                                item + feat_idx_dict['item']]])
                features_j = torch.LongTensor([[user + feat_idx_dict['user'], 
                                                item + feat_idx_dict['item']]])
                feature_values_i = torch.FloatTensor([[1, 1]])
                feature_values_j = torch.FloatTensor([[1, 1]])

                if torch.cuda.is_available():
                    features_i = features_i.cuda()
                    feature_values_i = feature_values_i.cuda()
                    features_j = features_j.cuda()
                    feature_values_j = feature_values_j.cuda()
                else:
                    features_i = features_i.cpu()
                    feature_values_i = feature_values_i.cpu()
                    features_j = features_j.cpu()
                    feature_values_j = feature_values_j.cpu()

                prediction_i, _ = model(features_i, feature_values_i, features_j, feature_values_j)
                prediction_i = prediction_i.clamp(min=-1.0, max=1.0)
                pred_rates.append(prediction_i[0].cpu().detach().item())

            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(items)[rec_idx]
            preds[user] = list(top_n)

        for u in preds.keys():
            preds[u] = [1 if e in test_ur[u] else 0 for e in preds[u]]

        precision_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
        fnl_precision.append(precision_k)

        recall_k = np.mean([recall_at_k(r, len(test_ur[u]), args.topk) for u, r in preds.items()])
        fnl_recall.append(recall_k)

        map_k = map_at_k(list(preds.values()))
        fnl_map.append(map_k)

        ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])
        fnl_ndcg.append(ndcg_k)

        hr_k = hr_at_k(list(preds.values()), list(preds.keys()), test_ur)
        fnl_hr.append(hr_k)

        mrr_k = mrr_at_k(list(preds.values()))
        fnl_mrr.append(mrr_k)

    print('---------------------------------')
    print(f'Precision@{args.topk}: {np.mean(fnl_precision)}')
    print(f'Recall@{args.topk}: {np.mean(fnl_recall)}')
    print(f'MAP@{args.topk}: {np.mean(fnl_map)}')
    print(f'NDCG@{args.topk}: {np.mean(fnl_ndcg)}')
    print(f'HR@{args.topk}: {np.mean(fnl_hr)}')
    print(f'MRR@{args.topk}: {np.mean(fnl_mrr)}')
