#!/usr/bin/env python3
"""
Gradients for the LambdaRank implementation.

.. moduleauthor:: Nishant Sharma
"""

import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

if os.name == "posix":
    libLambdaRankPath = os.path.dirname(__file__) + "/liblambdarank.so"
elif os.name == "nt":
    libLambdaRankPath = os.path.dirname(__file__) + "/liblambdarank.dll"
else:
    libLambdaRankPath = None

if libLambdaRankPath and os.path.exists(libLambdaRankPath):
    lambdarank_module = tf.load_op_library(libLambdaRankPath)

@ops.RegisterGradient("LambdaRank")
def LambdaRankGradient(op,
                       grad_ranknet_cost,
                       grad_lambdarank_cost,
                       grad_discrete_metric,
                       grad_lambdas):
    """
    The gradient for `lambdarank` using the operation implemented in C++.

    :param op: `lambdarank` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `lambdarank` op.
    :return: gradients with respect to the input of `lambdarank`.

    PLEASE NOTE: Although, the OP has other outputs, the differentiation works only
    w.r.t. lambdarank_score.
    """
    lambdas = op.outputs[3]
    grad_wrt_ypred = tf.multiply(grad_lambdarank_cost, lambdas)
    return [None, None, grad_wrt_ypred]

class LambdaRankLoss(object):
    def __init__(self, max_k=20):
        self.max_k = max_k

    def __call__(self, y, y_pred):
        _qid = tf.to_int32(y[...,1])
        _y = tf.to_int32(y[...,0])
        _y_pred = y_pred[...,0]
        lambdaRankNode = lambdarank_module.lambda_rank(_qid, _y, _y_pred, self.max_k)
        return lambdaRankNode.lambdarank_cost
