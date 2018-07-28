#include <vector>
#include <cmath>

using namespace std;

/*
    Class to efficiently compute NDCG scores.
    Also, fast implementation of delta NDCG. That means quickly finding out of
    change in NDCG score if two indices are swapped.
*/
class RunningNDCG
{
    int max_k;
    vector<double> discounts;
    vector<double> targets;
    double ideal_dcg, cur_dcg;
public:
    /*
    Constructor:

    Params:
        max_k determines the number of top ranking results that a user looks at.
    */
    RunningNDCG(int max_k):discounts(max_k)
    {
        this->max_k = max_k;
        for (int i = 0; i < max_k; i++)
        {
            discounts[i] = 1 / log2(double(i + 2));
        }
    }

    /*
    Init: This function loads target scores of an input ranking. It returns 
        the NDCG score of the given ranking. In addition, all subsequent swap_deltas
        work on this base ranking.

    Params:
        Iterator range to pick up target scores from.

    Returns:
        The NDCG score of the given ranking.

    */
    template<class TargetIterator>
    double init(TargetIterator targetsBegin, TargetIterator targetsEnd)
    {
        // Save target scores into this->targets.
        this->targets.clear();
        auto tIter = targetsBegin;
        while(tIter != targetsEnd)
        {
            this->targets.push_back(pow(2, *tIter)-1);
            tIter++;
        }

        // Sort by ranking targets to obtain ideal socre.
        vector<double> ideal_targets=targets;
        sort(
            ideal_targets.begin(),
            ideal_targets.end(),
            [](double x, double y) {return x>y; });

        // Calculate ideal DCG.
        ideal_dcg = 0;
        for (int i = 0; i < max_k; i++)
        {
            ideal_dcg += ideal_targets[i] * discounts[i];
        }
        if (abs(ideal_dcg) < 1e-10)
        {
            ideal_dcg = 1;
        }

        // Compute current DCG.
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

    /*
    * Change in score if we swap i and j.
    */
    double swap_delta(int i, int j) 
    {
        // Remove i&j contribution to DCG.
        double discount_i = (i >= max_k) ? 0 : discounts[i];
        double discount_j = (j >= max_k) ? 0 : discounts[j];
        double swapped_dcg_delta = (discount_j - discount_i) * (targets[i] - targets[j]);

        return swapped_dcg_delta / ideal_dcg;
    }

    /*
    NDCG score for an input target array. Only first max_k targets are picked.
    */
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


