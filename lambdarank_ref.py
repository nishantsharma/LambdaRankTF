import math

import numpy as np
import scipy

from pyltr.util.sort import get_sorted_y_positions
from pyltr.util.group import get_groups
from pyltr.metrics import NDCG

def query_lambdas(qid, y, y_pred, metric):
    """
    Reference version of which calculates the deltas slowly but easier to review.
    Used to make sure faster _calc_cross_lambdas method is correct.
    """
    n_samples = y.shape[0]
    currentRanking = list(get_sorted_y_positions(-y, y_pred, check=False))
    targetsForCurrentRanking = y[currentRanking]
    swap_deltas = metric.calc_swap_deltas(qid, targetsForCurrentRanking)

    discrete_metric = metric.evaluate(qid, targetsForCurrentRanking)
    print("P.DM ", discrete_metric)

    cross_lambdas = np.zeros((n_samples, n_samples))
    ranknet_cost = 0
    lambdarank_cost = 0
    for i in range(n_samples):
        for j in range(n_samples):
            i_rank_cur, j_rank_cur = currentRanking.index(i), currentRanking.index(j)
            if (min(i_rank_cur, j_rank_cur) > metric.max_k()):
                continue
            if (i == j):
                continue
            # print(i_rank_cur, j_rank_cur)
            if (y[i] > y[j]):
                S_ij = 1
            elif (y[i] < y[j]):
                S_ij = -1
            else:
                S_ij = 0

            delta_metric = swap_deltas[min(i_rank_cur,j_rank_cur), max(i_rank_cur,j_rank_cur)]
            if S_ij == 1:
                ranknet_cost_delta = math.log(1 + math.exp(y_pred[j]-y_pred[i]))
                cross_lambdas[i][j] = -abs(delta_metric)*scipy.special.expit(y_pred[j]-y_pred[i])
            elif S_ij == -1:
                ranknet_cost_delta = math.log(1 + math.exp(y_pred[i]-y_pred[j]))
                cross_lambdas[i][j] = abs(delta_metric)*scipy.special.expit(y_pred[i]-y_pred[j])
            else:
                ranknet_cost_delta = math.log(0.5*math.exp(y_pred[j]-y_pred[i]) + 0.5*math.exp(y_pred[i]-y_pred[j]))
                cross_lambdas[i][j] = -abs(delta_metric)*(0.5-scipy.special.expit(y_pred[j]-y_pred[i]))

            lambdarank_cost += ranknet_cost_delta * abs(delta_metric)
            ranknet_cost += ranknet_cost_delta
            # print(i, j, ranknet_cost_delta)

    # Cross entropy is double counted. Hence halving.
    lambdarank_cost /= 2
    ranknet_cost /= 2

    straight_lambdas = np.zeros((n_samples,))
    for i in range(n_samples):
        for j in range(n_samples):
            if (y[i] > y[j]):
                straight_lambdas[i] += cross_lambdas[i,j]
            elif (y[i] < y[j]):
                straight_lambdas[i] -= cross_lambdas[i,j]

    return ranknet_cost, discrete_metric, lambdarank_cost, straight_lambdas

def lambdarank_ref(qids, y, y_pred, metric):
    n_samples = y.shape[0]
    ranknet_cost = 0
    lambdarank_cost = 0
    discrete_metric = 0
    lambdas = np.zeros(qids.shape)
    metric = NDCG(k=7)
    for qid, a, b in get_groups(qids):
        (r, d, l, lambdas[a:b]) = query_lambdas(qid, y[a:b], y_pred[a:b], metric)
        ranknet_cost += r
        lambdarank_cost += l
        discrete_metric += d

    # print(ranknet_cost, lambdarank_cost, discrete_metric, lambdas)
    return (ranknet_cost, lambdarank_cost, discrete_metric, lambdas)