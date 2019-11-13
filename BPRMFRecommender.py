'''
@Author: Yu Di
@Date: 2019-09-29 10:56:31
@LastEditors: Yudi
@LastEditTime: 2019-11-13 15:14:40
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: BPR recommender
'''
import os
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from util.data_loader import BPRData, load_mat
from util.metrics import metric_eval, precision_at_k, recall_at_k, map_at_k, ndcg_at_k, hr_at_k, mrr_at_k

# model
class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        '''
        user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
        '''
        super(BPR, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        pred_i = (user * item_i).sum(dim=-1)
        pred_j = (user * item_j).sum(dim=-1)

        return pred_i, pred_j

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.01, 
                        help='learning rate')
    parser.add_argument('--wd', 
                        type=float, 
                        default=0.001, 
                        help='model regularization rate')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=4096, 
                        help='batch size for training')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help='training epochs')
    parser.add_argument('--top_k', 
                        type=int, 
                        default=10, 
                        help='compute metrics@K')
    parser.add_argument('--factor_num', 
                        type=int, 
                        default=32, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=4, 
                        help='sample negative items for training')
    parser.add_argument('--test_num_ng', 
                        type=int, 
                        default=99, 
                        help='sample part of negative items for testing')
    parser.add_argument('--out', 
                        default=True, 
                        help='save model or not')
    parser.add_argument('--gpu', 
                        default='0', 
                        help='gpu card ID')
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
                        default='cv', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    # load data
    src = args.dataset
    train_data_list, test_data, user_num, \
    item_num, train_mat_list, ur, val_data_list = load_mat(src, data_split=args.data_split, 
                                                           by_time=args.by_time, val_method=args.val_method, 
                                                           fold_num=args.fold_num)

    if args.val_method in ['tloo', 'loo', 'tfo']:
        fn = 1
    elif args.val_method == 'cv':
        fn = args.fold_num
    else:
        raise ValueError('Invalid val_method value')

    fnl_precision, fnl_recall, fnl_map, fnl_ndcg, fnl_hr, fnl_mrr = [], [], [], [], [], []
    for fold in range(fn):
        print(f'Start train Validation [{fold + 1}]......')
        train_dataset = BPRData(train_data_list[fold], item_num, train_mat_list[fold], args.num_ng, True)
        test_dataset = BPRData(test_data, item_num, train_mat_list[fold], 0, False)
        val_dataset = BPRData(val_data_list[fold], item_num, train_mat_list[fold], 0, False)
        
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                       shuffle=True, num_workers=4)
        test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, 
                                      shuffle=False, num_workers=0)
        val_loader = data.DataLoader(val_dataset, batch_size=args.test_num_ng + 1, 
                                      shuffle=False, num_workers=0)

        model = BPR(user_num, item_num, args.factor_num)
        if torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
        count, best_hr = 0, 0

        for epoch in range(args.epochs):
            model.train()
            start_time = time.time()
            train_loader.dataset.ng_sample()

            for user, item_i, item_j in train_loader:
                if torch.cuda.is_available():
                    user = user.cuda()
                    item_i = item_i.cuda()
                    item_j = item_j.cuda()
                else:
                    user = user.cpu()
                    item_i = item_i.cpu()
                    item_j = item_j.cpu()

                model.zero_grad()
                pred_i, pred_j = model(user, item_i, item_j)
                loss = -(pred_i - pred_j).sigmoid().log().sum()
                loss.backward()
                optimizer.step()

                count += 1

            model.eval()
            HR, NDCG = metric_eval(model, val_loader, args.top_k)

            elapsed_time = time.time() - start_time
            print('The time elapse of epoch {:03d}'.format(epoch + 1) + ' is: ' + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
            # print("HR: {:.3f}\tNDCG: {:.3f}}".format(np.mean(HR), np.mean(NDCG)))

            if HR > best_hr:
                best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
                if args.out:
                    if not os.path.exists(f'./models/{src}/'):
                        os.makedirs(f'./models/{src}/')
                    torch.save(model, f'./models/{src}/BPR.pt.{fold}')

        # calculate KPI
        preds = {}
        test_u_is = defaultdict(set)
        for ele in test_data:
            test_u_is[int(ele[0])].add(int(ele[1]))

        print('Start generate top-K rank list......')
        for u in tqdm(test_u_is.keys()):
            test_u_is[u] = list(test_u_is[u])
            pred_rates = [model(torch.tensor(u), torch.tensor(i), torch.tensor(i))[0].cpu().detach().item() for i in test_u_is[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.top_k]
            top_n = np.array(test_u_is[u])[rec_idx]
            preds[u] = list(top_n)

        for u in preds.keys():
            preds[u] = [1 if e in ur[u] else 0 for e in preds[u]]

        # calculate metrics
        precision_k = np.mean([precision_at_k(r, args.top_k) for r in preds.values()])
        fnl_precision.append(precision_k)

        recall_k = np.mean([recall_at_k(r, len(ur[u]), args.top_k) for u, r in preds.items()])
        fnl_recall.append(recall_k)

        map_k = map_at_k(list(preds.values()))
        fnl_map.append(map_k)

        ndcg_k = np.mean([ndcg_at_k(r, args.top_k) for r in preds.values()])
        fnl_ndcg.append(ndcg_k)

        hr_k = hr_at_k(list(preds.values()), list(preds.keys()), ur)
        fnl_hr.append(hr_k)

        mrr_k = mrr_at_k(list(preds.values()))
        fnl_mrr.append(mrr_k)

    print('---------------------------------')
    print(f'Precision@{args.top_k}: {np.mean(fnl_precision)}')
    print(f'Recall@{args.top_k}: {np.mean(fnl_recall)}')
    print(f'MAP@{args.top_k}: {np.mean(fnl_map)}')
    print(f'NDCG@{args.top_k}: {np.mean(fnl_ndcg)}')
    print(f'HR@{args.top_k}: {np.mean(fnl_hr)}')
    print(f'MRR@{args.top_k}: {np.mean(fnl_mrr)}')
