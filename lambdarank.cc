/// \file lambda_rank.cc
/// \author Nishant Sharma
/// \brief Implementation of a LambdaRank's cost calculations as an Op in Tensorflow.

#include <iostream>
#include <algorithm>
#include <vector>
#include <boost/range/irange.hpp>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/default/logging.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "RunningNDCG.h"

using namespace tensorflow;
using namespace boost;

/*
 * Function LambdaRank 
 *    Attributes:
 *        max_k: Maximum top rankers considered.
 *    Inputs:
 *        y: Array of (targetRating, qid).
 *        y_pred: Array of (computed Scores, qid).
 *    Outputs:
 *        ranknet_cost: Value of current Cross Entropy Cost "Function" as per RankNet algorithm.
 *        lambdarank_cost: Value of current Cross Entropy Cost "Function" as per LambdaRank algorithm.
 *                              Uses NDCG factors from current iteration.
 *        discrete_metric: Average NDCG score for each query in the set.
 *        lambdas: Gradients of lambdarank_cost w.r.t. predicted scores(y_pred).
 */
REGISTER_OP("LambdaRank")
  .Attr("max_k: int")
  .Input("qid: int32")
  .Input("y: float")
  .Input("y_pred: float")
  .Output("ranknet_cost: float")
  .Output("lambdarank_cost: float")
  .Output("discrete_metric: float")
  .Output("lambdas: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle qid_shape, y_shape, y_pred_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &qid_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &y_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &y_pred_shape));

    // Enforce shape for y_shape and y_pred_shape
    shape_inference::DimensionHandle n_samples = c->Dim(qid_shape, 0);
    TF_RETURN_IF_ERROR(c->Merge(qid_shape, c->Vector(n_samples), &qid_shape));
    TF_RETURN_IF_ERROR(c->Merge(y_shape, c->Vector(n_samples), &y_shape));
    TF_RETURN_IF_ERROR(c->Merge(y_pred_shape, c->Vector(n_samples), &y_pred_shape));

    // Extract max_k.
    int max_k = 0;
    c->GetAttr("max_k", &max_k);

    // Enforce output shapes.
    c->set_output(0, c->Scalar());
    c->set_output(1, c->Scalar());
    c->set_output(2, c->Scalar());
    c->set_output(3, c->Vector(n_samples));

    return Status::OK();
  });

/// \brief Implementation of an LambdaRank operation.
/// \param context
/// \author Nishant Sharma
class LambdaRankOp : public OpKernel {
    // Get the attributes.
    int max_k;
public:
    /// \brief Constructor.
    /// \param context
    explicit LambdaRankOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("max_k", &max_k));

        // Check that max_k is positive.
        OP_REQUIRES(context, max_k > 0, errors::InvalidArgument("Need preserve_index >= 0, got ", max_k));
    }

    /// \brief Compute the LambdaRank.
    /// \param context
    void Compute(OpKernelContext* context) override {
        cout << "\nStarted compute\n"; cout.flush();
        // some checks to be sure ...
        DCHECK_EQ(3, context->num_inputs());
        // Get the input tensors.
        const Tensor& qid = context->input(0);
        const Tensor& y = context->input(1);
        const Tensor& y_pred = context->input(2);

        // Check input dimension counts
        DCHECK_EQ(qid.shape().dims(), 1);
        DCHECK_EQ(y.shape().dims(), 1);
        DCHECK_EQ(y_pred.shape().dims(), 1);

        // Check dimensional compatibility.
        int n_samples = qid.shape().dim_size(0);
        int max_k = (this->max_k > n_samples) ? n_samples : this->max_k;
        DCHECK_EQ(y.dim_size(0), n_samples);
        DCHECK_EQ(y_pred.dim_size(0), n_samples);

        // create output shapes
        TensorShape lambdas_shape;
        lambdas_shape.AddDim(n_samples);
        TensorShape scalar_shape;

        // create output tensors
        Tensor* lambdas = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, lambdas_shape, &lambdas));

        Tensor *ranknet_cost = NULL;
        Tensor *lambdarank_cost = NULL, *discrete_metric = NULL;
        Tensor *cost_factors_curr = NULL, *lambdarank_cost_prev_delta = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, scalar_shape, &ranknet_cost));
        OP_REQUIRES_OK(context, context->allocate_output(0, scalar_shape, &lambdarank_cost));
        OP_REQUIRES_OK(context, context->allocate_output(0, scalar_shape, &discrete_metric));

        // Eigen tensors for input data access
        auto _qid = qid.vec<int32>();
        auto _y = y.vec<float>();
        auto _y_pred = y_pred.vec<float>();
        cout << "Moved compute              | | | | | | | | | | | | |\n"; cout.flush();
        cout << _qid; cout.flush();
        cout << "Printed stuff              | | | | | | | | | | | | |\n"; cout.flush();

        // Eigen tensors for output data access
        auto _lambdas = lambdas->vec<float>();
        auto _ranknet_cost = ranknet_cost->scalar<float>();
        auto _lambdarank_cost = lambdarank_cost->scalar<float>();
        auto _discrete_metric = discrete_metric->scalar<float>();

        // Initialize scores.
        _ranknet_cost() = 0;
        _lambdarank_cost() = 0;
        _discrete_metric() = 0;

        // Initialize lambdas.
        for (int i = 0; i < n_samples + 1; i++)
        {
            _lambdas(i) = 0;
        }

        // Used to calcualte NDCG and delta-NDCG.
        RunningNDCG runningNDCG(max_k);

        int a = 0, b = -1;
        int num_queries = 0;
        do
        {
            // a marks the first samples for query _qid[a].
            // Find b, such that [a, b) is an closed-open interval containing all samples
            // of query ID same as _qid[a].
            b = a;
            cout << "\nLoop starts with qids" << _qid(0) << "," << _qid(1) << endl; cout.flush();
            while (b != n_samples && _qid(a) == _qid(b))
            {
                b++;
                cout << _qid(b); cout.flush();
            }

            // Number of samples in the current query.
            int n_query_samples = b - a;

            cout << "Current query range "<<a<<", "<<b<<endl; cout.flush();
            // Increment number of queries found.
            num_queries++;

            // Obtain result ranking for the current query according to input y_pred's.
            auto query_span = irange(a, b);
            std::vector<int> currentRankingWithinQuery(query_span.begin(), query_span.end());
            std::sort(
                currentRankingWithinQuery.begin(),
                currentRankingWithinQuery.end(),
                [&_y, &_y_pred](int i, int j) {
                return (_y_pred(i) > _y_pred(j))
                    || (_y_pred(i) == _y_pred(j) && _y(i) < _y(j));
            }
            );
            cout << "After sorting" << endl; cout.flush();

            // targetsForCurrentRanking = y[currentRankingWithinQuery]
            std::vector<float> targetsForCurrentRanking(b - a);
            std::transform(currentRankingWithinQuery.begin(),
                currentRankingWithinQuery.end(),
                targetsForCurrentRanking.begin(),
                [&_y](int i) {return _y(i); });

            cout << "After transform." << endl; cout.flush();
            float _cur_discrete_metric = runningNDCG.init(
                targetsForCurrentRanking.begin(),
                targetsForCurrentRanking.end());
            _discrete_metric() += _cur_discrete_metric;

            float _lambdarank_cost_query = 0;
            float _ranknet_cost_query = 0;
            cout << "Started computing lambdas" << endl; cout.flush();
            for (int i_rank_cur = 0; i_rank_cur < max_k; i_rank_cur++)
            {
                int i = currentRankingWithinQuery[i_rank_cur];
                cout << i << "," << i_rank_cur << ".";
                for (int j_rank_cur = i_rank_cur + 1; j_rank_cur != n_query_samples; j_rank_cur++)
                {
                    int j = currentRankingWithinQuery[j_rank_cur];
                    cout << j << "," << j_rank_cur << ".";
                    float abs_swap_delta_ij = abs(runningNDCG.swap_delta(i_rank_cur, j_rank_cur));
                    float basic_value = 0;
                    float cross_lambda_ij = 0;

                    if (_y(i) > _y(j))
                    {
                        cross_lambda_ij = -abs_swap_delta_ij / (1 + exp(_y_pred(i) - _y_pred(j)));

                        _lambdas(i) += cross_lambda_ij;
                        _lambdas(j) += cross_lambda_ij;
#if 0
                        if (i == debugIndex)
                        {
                            print("[A]Adding {0}".format(cross_lambda_ij));
                        }
                        else if (j == debugIndex)
                        {
                            print("[B]Adding {0}".format(-cross_lambda_ij));
                        }
#endif

                        basic_value = log(1 + exp(_y_pred(j) - _y_pred(i)));
                    }
                    else if (_y(i) < _y(j))
                    {
                        cross_lambda_ij = abs_swap_delta_ij / (1 + exp(_y_pred(j) - _y_pred(i)));

                        _lambdas(i) -= cross_lambda_ij;
                        _lambdas(j) -= cross_lambda_ij;
#if 0
                        if (i == debugIndex)
                        {
                            print("[C]Adding {0}".format(-cross_lambda_ij));
                        }
                        else if (j == debugIndex)
                        {
                            print("[D]Adding {0}".format(cross_lambda_ij));
                        }
#endif

                        basic_value = log(1 + exp(_y_pred(i) - _y_pred(j)));
                    }
                    else
                    {
                        basic_value = log(0.5*exp(_y_pred(j) - _y_pred(i))
                            + 0.5*exp(_y_pred(i) - _y_pred(j)));
                    }

                    _lambdarank_cost_query += abs_swap_delta_ij * basic_value;
                    _ranknet_cost_query += basic_value;
                }

                _lambdarank_cost() += _lambdarank_cost_query;
                _ranknet_cost() += _ranknet_cost_query;
            }
            cout << "Done computing lambdas" << endl; cout.flush();
        } while (b != n_samples);

        _ranknet_cost() /= num_queries;
        _lambdarank_cost() /= num_queries;
        _discrete_metric() /= num_queries;
        cout << "Done compute            | | | | | | | | | | | | |\n"; cout.flush();
        cout << _ranknet_cost() << ", " << _lambdarank_cost() << ", " << _discrete_metric() << endl;
    }
};

REGISTER_KERNEL_BUILDER(Name("LambdaRank").Device(DEVICE_CPU), LambdaRankOp);
