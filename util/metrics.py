'''
@Author: Yu Di
@Date: 2019-09-29 13:41:24
@LastEditors: Yudi
@LastEditTime: 2019-09-29 14:18:59
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: metrics for top-N recommendation results
'''
import numpy as np

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
    return np.mean(r)

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

def mean_average_precision(rs):
    '''
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    '''
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Discounted cumulative gain
    '''
    r = np.asfarray(r)[:k]
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

