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

def runTest1():
    inputs = generateInputs()

    # Call python version.
    ref_values = lambdarank_ref(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
    (ranknet_cost_p, lambdarank_cost_p, discrete_metric_p, lambdas_p) = ref_values

    # Call tensorflow op version.
    with tf.Session('') as sess:
        # Create placeholders for TF OP inputs.
        _qids = tf.placeholder(tf.int32, shape = (inputs.n_samples,))
        _y = tf.placeholder(tf.float64, shape = (inputs.n_samples,))
        _y_pred = tf.placeholder(tf.float64, shape = (inputs.n_samples,))

        # Build feed_dict and feed it to TF session to obtain OP output.
        feed_dict = {_qids: inputs.qids, _y:inputs.y, _y_pred:inputs.y_pred}
        tf_value_nodes = lambdarank_module.lambda_rank(_qids, _y, _y_pred, inputs.max_k)

        # Obtain TF OP output.
        tf_values = sess.run(tf_value_nodes, feed_dict=feed_dict)
        (ranknet_cost_tf, lambdarank_cost_tf, discrete_metric_tf, lambdas_tf) = tf_values

        # Match TF OP output with reference output.
        # assert(abs(ranknet_cost_p - ranknet_cost_tf) < 1e-5)
        assert(abs(discrete_metric_p - discrete_metric_tf) < 1e-5 * (discrete_metric_p + discrete_metric_tf))
        assert(abs(lambdarank_cost_p - lambdarank_cost_tf) < 1e-5 * (lambdarank_cost_p + lambdarank_cost_tf))
        assert(np.linalg.norm(lambdas_p - lambdas_tf) < 1e-5)

def runTest2():
    inputs = generateInputs()

    # Call python version.
    ref_values = lambdarank_ref(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
    (ranknet_cost_p, lambdarank_cost_p, discrete_metric_p, lambdas_p) = ref_values

    delta = 0.000001

    lambdarank_cost_delta = np.zeros((inputs.n_samples, 2))
    for i in range(inputs.n_samples):
        oldValue = inputs.y_pred[i]

        inputs.y_pred[i] = oldValue + delta
        ref_values = lambdarank_ref(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
        (_, lambdarank_cost_n, _, _) = ref_values
        lambdarank_cost_delta[i][0] = (lambdarank_cost_n - lambdarank_cost_p) / delta

        inputs.y_pred[i] = oldValue - delta
        ref_values = lambdarank_ref(inputs.qids, inputs.y, inputs.y_pred, inputs.max_k)
        (_, lambdarank_cost_n, _, _) = ref_values
        lambdarank_cost_delta[i][1] = (lambdarank_cost_n - lambdarank_cost_p) / (-delta)

        inputs.y_pred[i] = oldValue

    import pdb;pdb.set_trace()

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
    runTest1()
    runTest2()
    # runTest3()
