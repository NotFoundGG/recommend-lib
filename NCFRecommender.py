'''
@Author: Yu Di
@Date: 2019-09-30 11:45:35
@LastEditors: Yudi
@LastEditTime: 2019-11-04 14:52:08
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Neural Collaborative Filtering Recommender
'''
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from util.data_loader import NCFData, load_mat
from util.metrics import metric_eval, recall_at_k, precision_at_k, map_at_k, ndcg_at_k, hr_at_k, mrr_at_k

class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout, 
                 model, GMF_model=None, MLP_model=None):
        '''
        user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
        '''
        super(NCF, self).__init__()
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)

        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        '''weights initialization'''
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                     a=1, nonlinearity='sigmoid')
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([self.GMF_model.predict_layer.weight, 
                                        self.MLP_model.predict_layer.weight], dim=1)
            predict_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.weight.data.copy_(0.5 * predict_bias)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), dim=-1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001, 
                        help='learning rate')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.0, 
                        help='dropout rate')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256, 
                        help='batch size for training')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help='training epochs')
    parser.add_argument('--top_k', 
                        type=int, 
                        default=10, 
                        help='compute metrics@top_k')
    parser.add_argument('--factor_num', 
                        type=int, 
                        default=32, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=3, 
                        help='number of layers in MLP model')
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=4, 
                        help='Sample negative items for training')
    parser.add_argument('--test_num_ng', 
                        type=int, 
                        default=99, 
                        help='sample part of negative items for testing')
    parser.add_argument('--out', 
                        default=True, 
                        help='save model or not')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    parser.add_argument('--model_name', 
                        type=str, 
                        default='NeuMF-end', 
                        help='target model name, if NeuMF-pre plz run MLP and GMF before')
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
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    # load data
    src = args.dataset
    train_data, test_data, user_num, item_num, train_mat, ur = load_mat(src, data_split=args.data_split, by_time=args.by_time)
    
    train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, 
                                  shuffle=False, num_workers=0)
    
    # model name 
    model_name = args.model_name
    assert model_name in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

    GMF_model_path = f'./models/{src}/GMF.pt'
    MLP_model_path = f'./models/{src}/MLP.pt'
    NeuMF_model_path = f'./models/{src}/NeuMF.pt'

    if model_name == 'NeuMF-pre':
        assert os.path.exists(GMF_model_path), 'lack of GMF model'    
        assert os.path.exists(MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(GMF_model_path)
        MLP_model = torch.load(MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, 
                model_name, GMF_model, MLP_model)

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    loss_function = nn.BCEWithLogitsLoss()
    
    if model_name == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    count, best_hr = 0, 0
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            if torch.cuda.is_available():
                user = user.cuda()
                item = item.cuda()
                label = label.float().cuda()
            else:
                user = user.cpu()
                item = item.cpu()
                label = label.float().cpu()

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            count += 1

        model.eval()
        HR, NDCG = metric_eval(model, test_loader, args.top_k, algo='ncf')
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch + 1) + ' is: ' + 
                time.strftime('%H: %M: %S', time.gmtime(elapsed_time)))
        # print('HR: {:.3f}\tNDCG: {:.3f}'.format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(f'./models/{src}'):
                    os.makedirs(f'./models/{src}')
                torch.save(model, f'./models/{src}/{model_name.split("-")[0]}.pt')

    print('End. Best epoch {:03d}: HR = {:.3f}'.format(best_epoch, best_hr))

    # Calculate KPI
    preds = {}
    test_u_is = defaultdict(set)
    for ele in test_data:
        test_u_is[int(ele[0])].add(int(ele[1]))

    print('Start generate top-K rank list......')
    for u in tqdm(test_u_is.keys()):
        test_u_is[u] = list(test_u_is[u])
        pred_rates = [model(torch.tensor(u), torch.tensor(i)).cpu().detach().numpy()[0] for i in test_u_is[u]]
        rec_idx = np.argsort(pred_rates)[::-1][:args.top_k]
        top_n = np.array(test_u_is[u])[rec_idx]
        preds[u] = list(top_n)

    for u in preds.keys():
        preds[u] = [1 if e in ur[u] else 0 for e in preds[u]]

    # calculate metrics
    precision_k = np.mean([precision_at_k(r, args.top_k) for r in preds.values()])
    print(f'Precision@{args.top_k}: {precision_k}')

    recall_k = np.mean([recall_at_k(r, len(ur[u]), args.top_k) for u, r in preds.items()])
    print(f'Recall@{args.top_k}: {recall_k}')

    map_k = map_at_k(list(preds.values()))
    print(f'MAP@{args.top_k}: {map_k}')

    ndcg_k = np.mean([ndcg_at_k(r, args.top_k) for r in preds.values()])
    print(f'NDCG@{args.top_k}: {ndcg_k}')

    hr_k = hr_at_k(list(preds.values()))
    print(f'HR@{args.top_k}: {hr_k}')

    mrr_k = mrr_at_k(list(preds.values()))
    print(f'MRR@{args.top_k}: {mrr_k}')
    