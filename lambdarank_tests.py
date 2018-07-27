#!/usr/bin/env python3
"""
Tests for LambdaRank Tensorflow operation.

.. moduleauthor:: Nishant Sharma()
"""

from attrdict import AttrDict
import unittest
import numpy as np
import tensorflow as tf
from lambdarank import *
from lambdarank_ref import *
from testModel import Slide2VecScoringModel

def generateInputs():
    numSlides = 10
    numQueries = 5
    wordVecLenIn=300
    wordVecLenReduced=50
    slideVecLen=50
    input_length=1
    maxRating = numSlides/2
    max_k = 7
    n_samples = numSlides * numQueries
    X = np.random.rand(numSlides * numQueries, wordVecLenIn)
    y = np.random.randint(0, maxRating+1, n_samples)
    rids = np.array([int(i%numSlides) for i in range(n_samples)])
    qids =  np.array([int(i/numSlides) for i in range(n_samples)])
    y_pred = np.random.rand(n_samples)
    return AttrDict(locals())

def lambdarank_tf(qids, y, y_pred, max_k):
    n_samples = len(y)
    _qids = tf.placeholder(tf.int32, shape = (n_samples,))
    _y = tf.placeholder(tf.float64, shape = (n_samples,))
    _y_pred = tf.placeholder(tf.float64, shape = (n_samples,))

    # Build feed_dict and feed it to TF session to obtain OP output.
    feed_dict = {_qids: qids, _y:y, _y_pred:y_pred}
    tf_value_nodes = lambdarank_module.lambda_rank(_qids, _y, _y_pred, max_k)

    # Obtain TF OP output.
    with tf.Session('') as sess:
        tf_values = sess.run(tf_value_nodes, feed_dict=feed_dict)

    return tf_values

def runTest1():
    for i in range(10):
        np.random.seed(i)
        inputs = generateInputs()

        # Call python version.
        ref_values = lambdarank_ref(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
        (ranknet_cost_ref, lambdarank_cost_ref, discrete_metric_ref, lambdas_ref) = ref_values

        # Call tensorflow op version.
        tf_values = lambdarank_tf(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
        (ranknet_cost_tf, lambdarank_cost_tf, discrete_metric_tf, lambdas_tf) = tf_values

        # Match TF OP output with reference output.
        # assert(abs(ranknet_cost_ref - ranknet_cost_tf) < 1e-5)
        assert(abs(discrete_metric_ref - discrete_metric_tf) < 1e-5 * (discrete_metric_ref + discrete_metric_tf))
        assert(abs(lambdarank_cost_ref - lambdarank_cost_tf) < 1e-5 * (lambdarank_cost_ref + lambdarank_cost_tf))
        # import pdb;pdb.set_trace()
        assert(np.linalg.norm(lambdas_ref - lambdas_tf) < 1e-5)

#def runTest2A():
#    inputs = generateInputs()

#    from pyltr.metrics import NDCG
#    from pyltr.util.group import get_groups
#    metric = NDCG(k=7)
#    delta = 0.000001

#    cost_grad_ref = np.zeros((inputs.n_samples))
#    cost_grad_tf = np.zeros((inputs.n_samples))

#    # Call python version.
#    ref_values = lambdarank_ref(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
#    (_, cost_p, _, lambdas_p) = ref_values
#    for qid, a, b in get_groups(qids):
#        (r, l, d, lambdas[a:b], _, _, p) = query_lambdas(qid, y[a:b], y_pred[a:b], metric)

#        for i in range(a, b):
#            # Change y_pred[i]
#            oldValue = inputs.y_pred[i]
#            inputs.y_pred[i] = oldValue + delta

#            # Call reference version.
#            ref_values = lambdarank_ref(inputs.qids, inputs.y, inputs.y_pred, metric)
#            (_, cost_ref, _, _) = ref_values
#            cost_grad_ref[i] = (cost_ref-cost_p)/delta

#            # Call tensorflow op version.
#            tf_values = lambdarank_tf(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
#            (_, cost_tf, _, _) = tf_values
#            cost_grad_tf[i] = (cost_tf-cost_p)/delta

#    import pdb;pdb.set_trace()
#    assert(np.linalg.norm(lambdas_p - cost_grad_ref) < 1e-5)
#    assert(np.linalg.norm(lambdas_p - cost_grad_tf) < 1e-5)

def runTest2():
    inputs = generateInputs()

    from pyltr.metrics import NDCG
    from pyltr.util.group import get_groups
    metric = NDCG(k=7)
    delta = 0.000001

    cost_grad_ref_emp = np.zeros((inputs.n_samples))
    cost_grad_tf_emp = np.zeros((inputs.n_samples))

    # Call python version.
    ref_values = lambdarank_ref(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
    (_, cost_p_ref, _, lambdas_p_ref) = ref_values

    # Call tensorflow OP version.
    tf_values = lambdarank_tf(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
    (_, cost_p_tf, _, lambdas_p_tf) = tf_values

    for i in range(inputs.n_samples):
        # Change y_pred[i]
        oldValue = inputs.y_pred[i]
        inputs.y_pred[i] = oldValue + delta

        # Call reference version.
        ref_values = lambdarank_ref(inputs.qids, inputs.y, inputs.y_pred, metric)
        (_, cost_ref, _, _) = ref_values
        cost_grad_ref_emp[i] = (cost_ref-cost_p_ref)/delta

        # Call tensorflow op version.
        tf_values = lambdarank_tf(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
        (_, cost_tf, _, _) = tf_values
        cost_grad_tf_emp[i] = (cost_tf-cost_p_tf)/delta

        inputs.y_pred[i] = oldValue

    assert(np.linalg.norm(lambdas_p_ref - lambdas_p_tf) < 1e-5)
    assert(np.linalg.norm(lambdas_p_ref - cost_grad_ref_emp) < 1e-5)
    assert(np.linalg.norm(lambdas_p_tf - cost_grad_tf_emp) < 1e-5)

def runTest3():
    inputs = generateInputs()

    scoringModel = Slide2VecScoringModel(
        wordVecLenIn=inputs.wordVecLenIn,
        wordVecLenReduced=inputs.wordVecLenReduced,
        slideVecLen=inputs.slideVecLen,
        slideCount=inputs.numSlides,
        input_length=inputs.input_length)

    # lambdaRankModel._calc_cross_lambdas(1.0, y, y_pred, debugIndex=-1, full=True, debug=True)
    scoringModel.fit(inputs.X, inputs.y, inputs.rids, inputs.qids, max_k=inputs.max_k)
    scoringModel.predict(inputs.X, inputs.rids, inputs.qids)

if __name__ == "__main__":
    runTest2()
    runTest1()
    runTest3()
