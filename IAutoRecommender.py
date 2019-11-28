'''
@Author: Yu Di
@Date: 2019-11-18 11:32:54
@LastEditors: Yudi
@LastEditTime: 2019-11-28 17:36:49
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: AutoEncoder for Recommender system
'''
# TODO change to pytorch version in future
import os
import time
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf

from util.data_loader import AutoRecData
from util.metrics import precision_at_k, recall_at_k, hr_at_k, map_at_k, mrr_at_k, ndcg_at_k

class AutoRec(object):
    def __init__(self, sess, args, num_users, num_items, R, mask_R, C, train_R, train_mask_R, 
                 test_R, test_mask_R,num_train_ratings,num_test_ratings, user_train_set, 
                 item_train_set, user_test_set, item_test_set):
        self.sess = sess
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        self.R = R
        self.mask_R = mask_R
        self.C = C
        self.train_R = train_R
        self.train_mask_R = train_mask_R
        self.test_R = test_R
        self.test_mask_R = test_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set

        self.hidden_neuron = args.hidden_neuron
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))

        self.base_lr = args.base_lr
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step
        self.random_seed = args.random_seed

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = args.decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf.compat.v1.train.exponential_decay(self.base_lr, self.global_step,
                                                       self.decay_step, 0.96, staircase=True)
        self.lambda_value = args.lambda_value

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []

        self.grad_clip = args.grad_clip

    def l2_norm(self,tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

    def run(self):
        self.prepare_model()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.train_epoch):
            self.train_model(epoch)
            self.test_model(epoch)

    def prepare_model(self):
        self.input_R = tf.compat.v1.placeholder(dtype=tf.float32, 
                                                shape=[None, self.num_items], 
                                                name='input_R')
        self.input_mask_R = tf.compat.v1.placeholder(dtype=tf.float32, 
                                                     shape=[None, self.num_items], 
                                                     name='input_mask_R')
        V = tf.compat.v1.get_variable(name='V', 
                                      initializer=tf.random.truncated_normal(shape=[self.num_items, 
                                                                                    self.hidden_neuron],
                                                                             mean=0, 
                                                                             stddev=0.03), 
                            dtype=tf.float32)
        W = tf.compat.v1.get_variable(name='W', 
                                      initializer=tf.random.truncated_normal(shape=[self.hidden_neuron, 
                                                                                    self.num_items],
                                                                             mean=0, 
                                                                             stddev=0.03),
                                      dtype=tf.float32)
        mu = tf.compat.v1.get_variable(name='mu', 
                                       initializer=tf.zeros(shape=self.hidden_neuron), 
                                       dtype=tf.float32)
        b = tf.compat.v1.get_variable(name='b', 
                                      initializer=tf.zeros(shape=self.num_items), 
                                      dtype=tf.float32)

        pre_Encoder = tf.matmul(self.input_R, V) + mu
        self.Encoder = tf.nn.sigmoid(pre_Encoder)
        pre_Decoder = tf.matmul(self.Encoder,W) + b
        self.Decoder = tf.identity(pre_Decoder)

        pre_rec_cost = tf.multiply((self.input_R - self.Decoder) , self.input_mask_R)
        rec_cost = tf.square(self.l2_norm(pre_rec_cost))
        pre_reg_cost = tf.square(self.l2_norm(W)) + tf.square(self.l2_norm(V))
        reg_cost = self.lambda_value * 0.5 * pre_reg_cost
        
        self.cost = rec_cost + reg_cost

        if self.optimizer_method == 'Adam':
            # optimizer = tf.train.AdamOptimizer(self.lr)
            optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == 'RMSProp':
            # optimizer = tf.train.RMSPropOptimizer(self.lr)
            optimizer = tf.compat.v1.train.RMSPropOptimizer(self.lr)
        else:
            raise ValueError('Optimizer Key Error')

        if self.grad_clip:
            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        else:
            self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

    def train_model(self, itr):
        start_time = time.time()
        random_perm_doc_idx = np.random.permutation(self.num_users)
        
        batch_cost = 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i + 1) * self.batch_size]
            
            _, cost = self.sess.run([self.optimizer, self.cost],
                                    feed_dict={self.input_R: self.train_R[batch_set_idx, :],
                                               self.input_mask_R: self.train_mask_R[batch_set_idx, :]})
            batch_cost += cost
        self.train_cost_list.append(batch_cost)

        if (itr + 1) % self.display_step == 0:
            print(f'Training Epoch [{itr + 1}]') 
            print('Total cost = {:.2f}'.format(batch_cost),
                  'Elapsed time : {:.2f} sec'.format(time.time() - start_time))

    def test_model(self, itr):
        start_time = time.time()
        cost, Decoder = self.sess.run([self.cost,self.Decoder],
                                      feed_dict={self.input_R: self.test_R, 
                                                 self.input_mask_R: self.test_mask_R})
        self.test_cost_list.append(cost)
        if (itr + 1) % self.display_step == 0:
            Estimated_R = Decoder.clip(min=1, max=5) # rating scale from 1-5
            unseen_user_test_list = list(self.user_test_set - self.user_train_set)
            unseen_item_test_list = list(self.item_test_set - self.item_train_set)

            for user in unseen_user_test_list:
                for item in unseen_item_test_list:
                    if self.test_mask_R[user,item] == 1: # exist in test set
                        Estimated_R[user,item] = 3
            
            pre_numerator = np.multiply((Estimated_R - self.test_R), self.test_mask_R)  # choose non-zero in test set
            numerator = np.sum(np.square(pre_numerator))
            denominator = self.num_test_ratings
            RMSE = np.sqrt(numerator / float(denominator))
            self.test_rmse_list.append(RMSE)
            print(f'Testing Epoch [{itr + 1}]')
            print ('Total cost = {:.2f}'.format(cost), 
                   'RMSE = {:.5f}'.format(RMSE),
                   'Elapsed time : {:.2f} sec'.format(time.time() - start_time))
            print ('=' * 50)
        
            self.prediction = Estimated_R

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoRec')
    # common settings
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
                        default='cv', 
                        help='validation method, options: cv, tfo, loo, tloo')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    # specific setting for autorec
    parser.add_argument('--hidden_neuron', type=int, default=500)
    parser.add_argument('--lambda_value', type=float, default=1)
    parser.add_argument('--train_epoch', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--optimizer_method', choices=['Adam','RMSProp'], default='Adam')
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--decay_epoch_step', 
                        type=int, 
                        default=50,
                        help="decay the learning rate for each n epochs")
    parser.add_argument('--random_seed', type=int, default=2019)
    parser.add_argument('--display_step', type=int, default=200)
    args = parser.parse_args()

    tf.compat.v1.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.val_method in ['tfo', 'tloo', 'lo']:
        fn = 1
    else:
        fn = args.fold_num

    data = AutoRecData(1, 0, args.dataset, args.prepro, args.data_split, 
                       args.by_time, args.val_method, args.fold_num)

    # calculate kpi
    fnl_precision, fnl_recall, fnl_map, fnl_ndcg, fnl_hr, fnl_mrr = [], [], [], [], [], []
    item_pool = list(range(data.item_num))
    max_i_num = 1000
    test_u_is = defaultdict(set)
    for key, val in data.test_ur.items():
        if len(val) < max_i_num:
            cands_num = max_i_num - len(val)
            sub_item_pool = set(item_pool) - set(data.train_ur[key]) - set(data.test_ur[key])
            cands = random.sample(sub_item_pool, cands_num)
            test_u_is[key] = set(data.test_ur[key]) | set(cands)
        else:
            test_u_is[key] = set(random.sample(val, max_i_num))
        test_u_is[key] = list(test_u_is[key])

    for fold in range(fn):
        print(f'Start Validation [{fold + 1}]......')
        config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth=True
        with tf.compat.v1.Session(config=config) as sess:
            algo = AutoRec(sess, args, data.user_num, data.item_num, data.R, data.mask_R, data.C, 
                        data.train_R[fold], data.train_mask_R[fold], data.test_R, data.test_mask_R, 
                        data.num_train_ratings[fold], data.num_test_ratings, data.user_train_set[fold], 
                        data.item_train_set[fold], data.user_test_set, data.item_test_set)
            algo.run()
    
        preds = {}
        for u in test_u_is.keys():
            pred_rates = [algo.prediction[u, i] for i in test_u_is[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
            top_n = np.array(test_u_is[u])[rec_idx]
            preds[u] = list(top_n)
            preds[u] = [1 if e in data.test_ur[u] else 0 for e in preds[u]]

        # calculate metrics
        precision_k = np.mean([precision_at_k(r, args.topk) for r in preds.values()])
        fnl_precision.append(precision_k)

        recall_k = np.mean([recall_at_k(r, len(data.test_ur[u]), args.topk) for u, r in preds.items()])
        fnl_recall.append(recall_k)

        map_k = map_at_k(list(preds.values()))
        fnl_map.append(map_k)

        ndcg_k = np.mean([ndcg_at_k(r, args.topk) for r in preds.values()])
        fnl_ndcg.append(ndcg_k)

        hr_k = hr_at_k(list(preds.values()), list(preds.keys()), data.test_ur)
        fnl_hr.append(hr_k)
        
        mrr_k = mrr_at_k(list(preds.values()))
        fnl_mrr.append(mrr_k) 

        tf.compat.v1.reset_default_graph()

    print('---------------------------------')
    print(f'Precision@{args.topk}: {np.mean(fnl_precision)}')
    print(f'Recall@{args.topk}: {np.mean(fnl_recall)}')
    print(f'MAP@{args.topk}: {np.mean(fnl_map)}')
    print(f'NDCG@{args.topk}: {np.mean(fnl_ndcg)}')
    print(f'HR@{args.topk}: {np.mean(fnl_hr)}')
    print(f'MRR@{args.topk}: {np.mean(fnl_mrr)}')