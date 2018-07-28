import math

import numpy as np
import scipy

from pyltr.util.sort import get_sorted_y_positions
from pyltr.util.group import get_groups
from pyltr.metrics import NDCG

def calc_lambdas(qid, y, y_pred, metric):
    """
    Reference version of LambdaRank algorithm, which calculates the gradients slowly
    but is easier to review. Used to make sure faster _calc_cross_lambdas method is
    correct.
    """
    n_samples = y.shape[0]
    currentRanking = list(get_sorted_y_positions(-y, y_pred, check=False))
    targetsForCurrentRanking = y[currentRanking]
    swap_deltas = metric.calc_swap_deltas(qid, targetsForCurrentRanking)

    discrete_metric = metric.evaluate(qid, targetsForCurrentRanking)

    cross_lambdas = np.zeros((n_samples, n_samples))
    lambdarank_cost_breakup = np.zeros((n_samples, n_samples))
    ranknet_cost = 0
    lambdarank_cost = 0
    pairCount = 0
    for i in range(n_samples):
        for j in range(n_samples):
            i_rank_cur, j_rank_cur = currentRanking.index(i), currentRanking.index(j)
            if (i == j):
                continue
            if (min(i_rank_cur, j_rank_cur) >= metric.max_k()):
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

            pairCount += 1
            lambdarank_cost += ranknet_cost_delta * abs(delta_metric)
            lambdarank_cost_breakup[i,j] = ranknet_cost_delta * abs(delta_metric)
            ranknet_cost += ranknet_cost_delta
            # print(i, j, ranknet_cost_delta)

    # Scaling down by 2, to take care of double counting.
    lambdarank_cost /= 2
    lambdarank_cost_breakup /= 2
    ranknet_cost /= 2

    straight_lambdas = np.zeros((n_samples,))
    for i in range(n_samples):
        for j in range(n_samples):
            straight_lambdas[i] += cross_lambdas[i,j]
            # if (y[i] > y[j]):
            #     straight_lambdas[i] += cross_lambdas[i,j]
            # elif (y[i] < y[j]):
            #     straight_lambdas[i] -= cross_lambdas[i,j]

    return (ranknet_cost, lambdarank_cost, discrete_metric, straight_lambdas,
            cross_lambdas, lambdarank_cost_breakup, pairCount)


def lambdarank_ref(qids, y, y_pred, metric):
    n_samples = y.shape[0]
    ranknet_cost = 0
    lambdarank_cost = 0
    discrete_metric = 0
    pairCount = 0
    lambdas = np.zeros(qids.shape)
    metric = NDCG(k=7)
    n_queries = 0
    for qid, a, b in get_groups(qids):
        (r, l, d, lambdas[a:b], _, _, p) = calc_lambdas(qid, y[a:b], y_pred[a:b], metric)
        ranknet_cost += r
        lambdarank_cost += l
        discrete_metric += d
        pairCount += p
        n_queries += 1

    # Scale by pairCount
    lambdarank_cost *= 100.0/n_samples
    ranknet_cost *= 100.0/n_samples
    discrete_metric /= n_queries
    lambdas *= 100.0/n_samples

    print(pairCount, n_queries, ranknet_cost, lambdarank_cost, discrete_metric)
    # print(lambdas)
    return (ranknet_cost, lambdarank_cost, discrete_metric, lambdas)