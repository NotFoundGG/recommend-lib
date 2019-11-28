'''
@Author: Yu Di
@Date: 2019-11-15 13:45:40
@LastEditors: Yudi
@LastEditTime: 2019-11-28 16:11:00
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
import os
import random
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util.data_loader import BuildCorpus, PermutedSubsampledCorpus, load_rate
from util.metrics import precision_at_k, recall_at_k, map_at_k, ndcg_at_k, hr_at_k, mrr_at_k

class Bundler(nn.Module):
    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError

class Item2Vec(Bundler):
    def __init__(self, vocab_size=20000, embedding_size=100, padding_idx=0):
        super(Item2Vec, self).__init__()
        self.vocab_size = vocab_size # item_num
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), 
                                                       torch.FloatTensor(self.vocab_size - 1, 
                                                       self.embedding_size).uniform_(-0.5 / self.embedding_size, 
                                                                                     0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), 
                                                       torch.FloatTensor(self.vocab_size - 1, 
                                                       self.embedding_size).uniform_(-0.5 / self.embedding_size, 
                                                                                     0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True
                                        
    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = torch.LongTensor(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = torch.LongTensor(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)

class SGNS(nn.Module):
    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = torch.FloatTensor(wf)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, 
                                       replacement=True).view(batch_size, -1)
        else:
            nwords = torch.FloatTensor(batch_size, 
                                       context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, 
                                                                             self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()

def cos_sim(a, b):
    numerator = np.multiply(a, b).sum()
    denomitor = np.linalg.norm(a) * np.linalg.norm(b)
    if denomitor == 0:
        return 0
    else:
        return numerator / denomitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepro', 
                        type=str, 
                        default='origin', 
                        help='dataset type for experiment, origin, 5core, 10core available')
    parser.add_argument('--topk', 
                        type=int, 
                        default=10, 
                        help='recommend number for rank list')
    parser.add_argument('--data_split', 
                        type=str, 
                        default='fo', 
                        help='method for split test,options: loo/fo')
    parser.add_argument('--by_time', 
                        type=int, 
                        default=0, 
                        help='whether split data by time stamp')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    parser.add_argument('--val_method', 
                        type=str, 
                        default='tloo', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    # item2vec settings
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./models/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=10, help="number of epochs") # 100
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    args = parser.parse_args()

    df = load_rate(args.dataset, args.prepro)
    # convert to int code for easy look
    df['user'] = pd.Categorical(df.user).codes
    df['item'] = pd.Categorical(df.item).codes

    pre = BuildCorpus(df, args.window, args.max_vocab, args.unk, args.dataset)
    pre.build()

    if args.data_split == 'fo':
        if args.by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)

            split_idx = int(np.ceil(len(df) * 0.8))
            train, test = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()
        else:
            train, test = train_test_split(df, test_size=.2)
    elif args.data_split == 'loo':
        if args.by_time:
            df = df.sample(frac=1)
            df = df.sort_values(['timestamp']).reset_index(drop=True)

            df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
            train, test = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
            del train['rank_latest'], test['rank_latest']
        else:
            test = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
            test_key = test[['user', 'item']].copy()
            train = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()
    else:
        raise ValueError('Invalid data_split value, expect: loo, fo')

    # build average vector for certain user
    train_ur = defaultdict(set)
    for _, row in train.iterrows():
        train_ur[int(row['user'])].add(int(row['item']))
    test_ur = defaultdict(set)
    for _, row in test.iterrows():
        test_ur[int(row['user'])].add(int(row['item']))

    test_u_is = defaultdict(set)
    max_i_num = 1000
    item_pool = list(range(df.item.nunique()))
    for key, val in test_ur.items():
        # build candidates set
        if len(val) < max_i_num:
            cands_num = max_i_num - len(val)
            sub_item_pool = list(set(item_pool) - train_ur[key] - test_ur[key])
            cands = random.sample(sub_item_pool, cands_num)
            test_u_is[key] = list(test_ur[key] | set(cands))
        else:
            test_u_is[key] = list(random.sample(val, max_i_num))

    train_list, val_list = [], []
    if args.val_method == 'cv':
        kf = KFold(n_splits=args.fold_num, shuffle=False, random_state=2019)
        for train_index, val_index in kf.split(train):
            train_list.append(train.iloc[train_index, :])
            val_list.append(train.iloc[val_index, :])
    elif args.val_method == 'loo':
        val_set = train.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        val_key = val_set[['user', 'item']].copy()
        train = train.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(val_key)).reset_index().copy()

        train_list.append(train)
        val_list.append(val_set)
    elif args.val_method == 'tloo':
        train = train.sample(frac=1)
        train = train.sort_values(['timestamp']).reset_index(drop=True)

        train['rank_latest'] = train.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        train_set = train[train['rank_latest'] > 1].copy()
        val_set = train[train['rank_latest'] == 1].copy()
        del train_set['rank_latest'], val_set['rank_latest']

        train_list.append(train_set)
        val_list.append(val_set)
    elif args.val_method == 'tfo':
        train = train.sample(frac=1)
        train = train.sort_values(['timestamp']).reset_index(drop=True)

        split_idx = int(np.ceil(len(train) * 0.9))
        train_set, val_set = train.iloc[:split_idx, :].copy(), train.iloc[split_idx:, :].copy()

        train_list.append(train_set)
        val_list.append(val_set)
    else:
        raise ValueError('Invalid val_method value, expect: cv, loo, tloo, tfo')

    if args.val_method in ['tloo', 'loo', 'tfo']:
        fn = 1
    else:
        fn = args.fold_num

    fnl_precision, fnl_recall, fnl_map, fnl_ndcg, fnl_hr, fnl_mrr = [], [], [], [], [], []
    for fold in range(fn):
        print(f'Start Validation [{fold + 1}]......')
        pre.convert(train_list[fold], fold)

        idx2item = pickle.load(open(os.path.join(args.data_dir, args.dataset, f'idx2item.dat'), 'rb'))
        wc = pickle.load(open(os.path.join(args.data_dir, args.dataset, f'wc.dat'), 'rb'))
        wf = np.array([wc[word] for word in idx2item])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(args.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        vocab_size = len(idx2item)
        weights = wf if args.weights else None
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        model = Item2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
        modelpath = os.path.join(args.save_dir, args.dataset, f'{args.name}.pt.{fold}')
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, weights=weights)
        if os.path.isfile(modelpath) and args.conti:
            sgns.load_state_dict(torch.load(modelpath))
        if args.cuda and torch.cuda.is_available():
            sgns = sgns.cuda()
        else:
            sgns = sgns.cpu()
        optimizer = optim.Adam(sgns.parameters())
        optimpath = os.path.join(args.save_dir, args.dataset, f'{args.name}.optimizer.pt.{fold}')
        if os.path.isfile(optimpath) and args.conti:
            optimizer.load_state_dict(torch.load(optimpath))
        for epoch in range(1, args.epoch + 1):
            dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, args.dataset, f'train.i2v.dat.{fold}'))
            dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
            total_batches = int(np.ceil(len(dataset) / args.mb))
            pbar = tqdm(dataloader)
            pbar.set_description(f'[Epoch {epoch}]')
            for iword, owords in pbar:
                loss = sgns(iword, owords)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
        idx2vec = model.ivectors.weight.data.cpu().numpy()
        pickle.dump(idx2vec, open(os.path.join(args.data_dir, args.dataset, f'idx2vec.dat.{fold}'), 'wb'))
        torch.save(sgns.state_dict(), os.path.join(args.save_dir, args.dataset, f'{args.name}.pt.{fold}'))
        torch.save(optimizer.state_dict(), os.path.join(args.save_dir, args.dataset, f'{args.name}.optimizer.pt.{fold}'))

        item2idx = pickle.load(open(os.path.join(args.data_dir, args.dataset, f'item2idx.dat'), 'rb'))
        item2vec = {key: idx2vec[val, :] for key, val in item2idx.items()}

        # calculate KPI
        print('---------------------------------')
        print('Start Calculating KPI......')
        test_uservec, preds = dict(), dict()
        for u in test_ur.keys():
            # calculate test user vector for similarity
            test_uservec[u] = np.array([item2vec[i] for i in test_ur[u]]).mean(axis=0)
            test_u_is[u] = list(test_u_is[u])
            pred_rates = [cos_sim(test_uservec[u], item2vec[i]) for i in test_u_is[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(test_u_is[u])[rec_idx]
            preds[u] = list(top_n)
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
