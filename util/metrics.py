'''
@Author: Yu Di
@Date: 2019-09-29 13:41:24
@LastEditors: Yudi
@LastEditTime: 2019-11-04 17:35:27
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: metrics for top-N recommendation results
'''
import numpy as np
import torch

# pytorch KPI calculating methods
# NFM train metric
def metrics_nfm(model, dataloader):
    # device = torch.device('cpu')
    RMSE = np.array([], dtype=np.float32)
    for features, feature_values, label in dataloader:
        if torch.cuda.is_available():
            features = features.cuda()
            feature_values = feature_values.cuda()
            label = label.cuda()
        else:
            features = features.cpu()
            feature_values = feature_values.cpu()
            label = label.cpu()

        prediction = model(features, feature_values)
        prediction = prediction.clamp(min=-1.0, max=1.0)
        SE = (prediction - label).pow(2)
        RMSE = np.append(RMSE, SE.detach().cpu().numpy())

    return np.sqrt(RMSE.mean())

def _hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

def _ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0

def _bpr_topk(model, test_loader, top_k):
    HR, NDCG = [], []
    for user, item_i, item_j in test_loader:
        if torch.cuda.is_available():
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda() # not useful when testing
        else:
            user = user.cpu()
            item_i = item_i.cpu()
            item_j = item_j.cpu()

        prediction_i, _ = model(user, item_i, item_j)
        _, indices = torch.topk(prediction_i, top_k)
        recommends = torch.take(item_i, indices).cpu().numpy().tolist()
        gt_item = item_i[0].item()
        
        HR.append(_hit(gt_item, recommends))
        NDCG.append(_ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

def _ncf_topk(model, test_loader, top_k):
    HR, NDCG = [], []
    for user, item, _ in test_loader: # _ is label
        if torch.cuda.is_available():
            user = user.cuda()
            item = item.cuda()
        else:
            user = user.cpu()
            item = item.cpu()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(_hit(gt_item, recommends))
        NDCG.append(_ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

def metric_eval(model, test_loader, top_k, algo='bpr'):
    if algo == 'bpr':
        HR, NDCG = _bpr_topk(model, test_loader, top_k)
    elif algo == 'ncf':
        HR, NDCG = _ncf_topk(model, test_loader, top_k)

    return HR, NDCG
############ these metric only work in train process ########################

# some algorithm just use numpy-based, so the KPI calculating methods are different from pytorch
# function below used for test set KPI calculation
def precision_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    '''
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    # return np.mean(r)
    return sum(r) / len(r)

def recall_at_k(r, groud_truth_len, k):
    if groud_truth_len != 0:
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        
        return sum(r) / groud_truth_len
    else:
        return 0

def mrr_at_k(rs):
    res = 0
    for r in rs:
        for index, item in enumerate(r):
            if item == 1:
                res += 1 / (index + 1)
    return res / len(rs)

def average_precision(r):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    '''
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / len(r)

def map_at_k(rs):
    '''
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    '''
    return np.mean([average_precision(r) for r in rs])

def hr_at_k(rs):
    return np.mean(rs)

def dcg_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Discounted cumulative gain
    '''
    r = np.asfarray(r)[:k] != 0
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Normalized discounted cumulative gain
    '''
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

