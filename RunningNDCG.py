import numpy as np
import math
from pyltr.util.sort import get_sorted_y_positions
from pyltr.util.group import get_groups

class RunningNDCG(object):
    def __init__(self, max_k):
        self.max_k = max_k
        self.discounts = np.array([1/math.log(i+2, 2) for i in range(max_k)])

    def init(self, targets):
        self.targets = np.array([math.pow(2, t)-1 for t in targets])
        self.ideal_dcg = np.dot(self.discounts, np.array(sorted(self.targets, key=lambda r:-r)[0:self.max_k]))
        if abs(self.ideal_dcg) < 1e-10:
            self.ideal_dcg = 1
        self.cur_dcg = np.dot(self.discounts, self.targets[0:self.max_k])
        return self.cur_dcg / self.ideal_dcg

    def swap_delta(self, i, j):
        # Remove i&j contribution to DCG.
        discount_i = 0 if i>=self.max_k else self.discounts[i]
        discount_j = 0 if j>=self.max_k else self.discounts[j]
        swapped_dcg_delta = (discount_j - discount_i) * (self.targets[i] - self.targets[j])

        return swapped_dcg_delta / self.ideal_dcg

    def calc(self, targets):
        targets = np.array([math.pow(2, t)-1 for t in targets])
        k = min(self.max_k, len(targets))
        cur_dcg = np.dot(self.discounts[0:k], targets[0:k])
        ideal_dcg = np.dot(self.discounts[0:k], np.array(sorted(targets, key=lambda r:-r)[0:k]))
        if abs(ideal_dcg) < 1e-10:
            ideal_dcg = 1
        return cur_dcg /ideal_dcg

    def calc_mean(self, qids, y, y_pred):
        count = 0
        sum = 0.
        for qid, a, b in get_groups(qids):
            currentRanking = list(get_sorted_y_positions(-y[a:b], y_pred[a:b]))
            targetsForCurrentRanking = y[a:b][currentRanking]
            sum += self.calc(targetsForCurrentRanking)
            count += 1
        return sum/count

    def calc_mean_random(self, qids, y):
        count = 0
        sum = 0.
        for qid, a, b in get_groups(qids):
            targetsForRandomRanking = np.random.permutation(y[a:b])
            sum += self.calc(targetsForRandomRanking)
            count += 1
        return sum/count
