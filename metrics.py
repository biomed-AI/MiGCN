import numpy as np
from functools import wraps
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import heapq
import torch
from utilty import *

import multiprocessing
cores = multiprocessing.cpu_count() // 2

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def logloss(ground_truth, prediction):
    # preds = [max(min(p, 1. - 10e-12), 10e-12) for p in prediction]
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    ret = auc(ground_truth=r, prediction=posterior)
    return ret


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    score = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
            score.append([i,item_score[i],-1])
        else:
            r.append(0)
            score.append([i,item_score[i]])
    auc = get_auc(item_score, user_pos_test)
    return r, auc, score

train_user_dict = dict()
test_user_dict = dict()
all_items = set()


def test_one_user(x):
    rating, user = x[0], x[1]
    try:
        train_items = train_user_dict[user]
    except Exception:
        train_items = []
    test_items = list(all_items- set(train_items))
    user_pos_test = test_user_dict[user]
    r, auc, score = ranklist_by_sorted(user_pos_test, test_items, rating, cmd_args.Ks)

    precision, recall, ndcg, hit_ratio = [], [], [], []
    for K in cmd_args.Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
        'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc, 'predict':[user] + score}


def batch_metrics(batch_predictions, user_batch, data_generator):
    global train_user_dict, test_user_dict, all_items
    train_user_dict = data_generator.train_user_dict
    test_user_dict = data_generator.test_user_dict
    all_items = set(range(data_generator.n_items))

    pool = multiprocessing.Pool(cores)
    user_batch_rating_uid = zip(batch_predictions, user_batch)
    batch_result = pool.map(test_one_user, user_batch_rating_uid)
    pool.close()

    return batch_result

