#include <vector>
#include <cmath>

using namespace std;

class RunningNDCG
{
    int max_k;
    vector<double> discounts;
    vector<double> targets;
    double ideal_dcg, cur_dcg;
public:
    RunningNDCG(int max_k):discounts(max_k)
    {
        this->max_k = max_k;
        for (int i = 0; i < max_k; i++)
        {
            discounts[i] = 1 / log2(double(i + 2));
        }
    }

    template<class TargetIterator>
    double init(TargetIterator targetsBegin, TargetIterator targetsEnd)
    {
        this->targets.clear();
        auto tIter = targetsBegin;
        while(tIter != targetsEnd)
        {
            this->targets.push_back(pow(2, *tIter)-1);
            tIter++;
        }

        vector<double> ideal_targets=targets;
        sort(
            ideal_targets.begin(),
            ideal_targets.end(),
            [](double x, double y) {return x>y; });

        ideal_dcg = 0; // This line is required to use calc correctly.
        for (int i = 0; i < max_k; i++)
        {
            ideal_dcg += ideal_targets[i] * discounts[i];
        }
        if (abs(ideal_dcg) < 1e-10)
        {
            ideal_dcg = 1;
        }

        cur_dcg = 0;
        for (int i = 0; i < max_k; i++)
        {
            cur_dcg += targets[i] * discounts[i];
        }
        if (abs(cur_dcg) < 1e-10)
        {
            cur_dcg = 1;
        }

        return cur_dcg / ideal_dcg;
    }

    double swap_delta(int i, int j) 
    {
        // Remove i&j contribution to DCG.
        double discount_i = (i >= max_k) ? 0 : discounts[i];
        double discount_j = (j >= max_k) ? 0 : discounts[j];
        double swapped_dcg_delta = (discount_j - discount_i) * (targets[i] - targets[j]);

        return swapped_dcg_delta / ideal_dcg;
    }

    template<typename T>
    double calc(T targets)
    {
        cur_dcg = 0;
        for (int i = 0; i < max_k; i++)
        {
            cur_dcg += (pow(2, targets[i]) - 1) * discounts[i];
        }
        if (abs(cur_dcg) < 1e-10)
        {
            cur_dcg = 1;
        }
        return cur_dcg / ideal_dcg;
    }

    void test()
    {
        /*
        # def unit_test(self):
        if debug :
        with blockProfiler("SwapDeltas") :
        swapDeltas = {}
        swap_deltas = self.metric.calc_swap_deltas(qid, targetsForCurrentRanking)
        for i_rank_cur, i in enumerate(currentRankingWithinQuery[0:max_k]) :
        for rank_gap, j in enumerate(currentRankingWithinQuery[i_rank_cur + 1:]) :
        j_rank_cur = rank_gap + i_rank_cur + 1
        if debug :
        assert(i_rank_cur == currentRankingWithinQuery.index(i))
        assert(j_rank_cur == currentRankingWithinQuery.index(j))

        # Swap i and j and check score again.
        whatIfRanking = np.copy(currentRankingWithinQuery)
        whatIfRanking[i_rank_cur], whatIfRanking[j_rank_cur] = whatIfRanking[j_rank_cur], whatIfRanking[i_rank_cur]
        targetsForWhatIfRanking = y[whatIfRanking]
        whatIfScore = self.metric.evaluate(qid, targetsForWhatIfRanking)
        swapDeltas[(i, j)] = whatIfScore - cur_discrete_metric

        if (abs(swapDeltas[(i, j)] - swap_deltas[i_rank_cur, j_rank_cur]) >= 1e-5) :
        import pdb; pdb.set_trace()
        self.metric.evaluate(qid, targetsForWhatIfRanking)
        self.runningNDCG.calc(targetsForWhatIfRanking)
        pass
        */
    }
};


