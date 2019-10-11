'''
@Author: Yu Di
@Date: 2019-09-30 15:27:46
@LastEditors: Yudi
@LastEditTime: 2019-10-11 18:47:34
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: Neural FM recommender
'''
import os
import time
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from util.metrics import metrics
from util.data_loader import load_libfm, map_features, FMData

class NFM(nn.Module):
    def __init__(self, num_features, num_factors, act_function, layers, batch_norm, drop_prob, pretrain_FM):
        '''
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        layers: list of dimension of deep layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        pretrain_FM: the pre-trained FM weights.
        '''
        super(NFM, self).__init__()
        self.num_features = num_features
        self.num_factors = num_factors
        self.act_function = act_function
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.pretrain_FM = pretrain_FM

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_module = []
        in_dim = num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

            if self.batch_norm:
                MLP_module.append(nn.BatchNorm1d(out_dim))
            if self.act_function == 'relu':
                MLP_module.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_module.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_module.append(nn.Tanh())

            MLP_module.append(nn.Dropout(drop_prob[-1]))
        self.deep_layers = nn.Sequential(*MLP_module)
        predict_size = layers[-1] if layers else num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weights_()

    def _init_weights_(self):
        '''
        Try to mimic the original weight initialization.
        '''
        if self.pretrain_FM:
            self.embeddings.weight.data.copy_(self.pretrain_FM.embeddings.weight)
            self.biases.weight.data.copy_(self.pretrain_FM.biases.weight)
            self.bias_.weight.data.copy_(self.pretrain_FM.bias_)
        else:
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.biases.weight, 0.0)

        # for deep layers
        if len(self.layers) > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM)
        if self.layers: # have deep layers
            FM = self.deep_layers(FM)
        FM = self.prediction(FM)

        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_

        return FM.view(-1)

class FM(nn.Module):
    def __init__(self, num_features, num_factors, batch_norm, drop_prob):
        '''
        num_features: number of features,
        num_factors: number of hidden factors,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        '''
        super(FM, self).__init__()
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

    def forward(self, features, feature_values):
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
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.05, 
                        help='learning rate')
    parser.add_argument('--dropout',
                        default='[0.5, 0.2]', 
                        help='dropout rate for FM and MLP')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128, 
                        help='batch size for training')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        help='training epochs')
    parser.add_argument('--hidden_factor', 
                        type=int, 
                        default=64, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--layers',
                        default='[64]', 
                        help='size of layers in MLP model, [] is NFM-0')
    parser.add_argument('--lamda', 
                        type=float, 
                        default=0.0, 
                        help='regularizer for bilinear layers')
    parser.add_argument('--batch_norm', 
                        default=True, 
                        help='use batch_norm or not')
    parser.add_argument('--pre_train', 
                        action='store_true', 
                        default=False, 
                        help='whether use the pre-train or not')
    parser.add_argument('--out', 
                        default=True, 
                        help='save model or not')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    parser.add_argument('--model', 
                        type=str, 
                        default='NFM',
                        help='model type, NFM or FM')
    parser.add_argument('--act_func', 
                        type=str, 
                        default='relu', 
                        help='type of activate function')
    parser.add_argument('--opt', 
                        type=str, 
                        default='Adagrad', 
                        help='type of optimizer')
    parser.add_argument('--crit', type=str, default='square_loss', help='square_loss or log_loss')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    # device = torch.device('cpu') # turn to cpu mode

    ### prepare dataset ###
    src = 'ml-100k'
    features_map, num_features = map_features(src)

    train_dataset = FMData(f'./data/{src}/{src}.train.libfm', features_map)
    valid_dataset = FMData(f'./data/{src}/{src}.valid.libfm', features_map)
    test_dataset = FMData(f'./data/{src}/{src}.test.libfm', features_map)

    train_loader = data.DataLoader(train_dataset, drop_last=True, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=4)
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ### create model ###
    model_path = './models/'
    assert args.model in ['FM', 'NFM'], 'invalid model type'

    opt = args.opt
    assert opt in ['Adagrad', 'Adam', 'SGD', 'Momentum']

    activation_function = args.act_func
    assert activation_function in ['relu', 'sigmoid', 'tanh', 'identity']

    assert args.crit in ['square_loss', 'log_loss']

    if args.pre_train:
        assert os.path.exists(model_path + 'FM.pt'), 'lack of FM model'
        assert args.model == 'NFM', 'only support NFM for now'
        FM_model = torch.load(model_path + 'FM.pt')
    else:
        FM_model = None

    if args.model == 'FM':
        model = FM(num_features, args.hidden_factor, args.batch_norm, eval(args.dropout))
    else:
        model = NFM(num_features, args.hidden_factor, activation_function, eval(args.layers), 
                    args.batch_norm, eval(args.dropout), FM_model)
    
    model.cuda()
    # model.to(device)

    if opt == 'Adagrad':
            optimizer = optim.Adagrad(
            model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
    elif opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif opt == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

    if args.crit == 'square_loss':
        criterion = nn.MSELoss(reduction='sum')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

    count, best_rmse = 0, 100
    for epoch in range(args.epochs):
        model.train() # Enable dropout and batch_norm
        start_time = time.time()
        
        for features, feature_values, label in train_loader:
            features = features.cuda()
            feature_values = feature_values.cuda()
            label = label.cuda()
            # features = features.to(device)
            # feature_values = feature_values.to(device)
            # label = label.to(device)

            model.zero_grad()
            prediction = model(features, feature_values)
            loss = criterion(prediction, label)
            loss += args.lamda * model.embeddings.weight.norm()
            loss.backward()
            optimizer.step()

            count += 1

        model.eval()
        train_result = metrics(model, train_loader)
        valid_result = metrics(model, valid_loader)
        test_result = metrics(model, test_loader)

        print('Runing Epoch {:03d} costs '.format(epoch) + time.strftime('%H: %M: %S', 
                                                                         time.gmtime(time.time() - start_time)))
        print('Train_RMSE: {:.3f}, Valid_RMSE: {:.3f}, Test_RMSE: {:.3f}'.format(train_result, 
                                                                                 valid_result, 
                                                                                 test_result))
        if test_result < best_rmse:
            best_rmse, best_epoch = test_result, epoch
            if args.out:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(model, f'{model_path}{src}/{args.model}.pt')
    print('End. Best epoch {:03d}: Test_RMSE is {:.3f}'.format(best_epoch, best_rmse))
    