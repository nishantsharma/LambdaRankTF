#!/usr/bin/env python3
"""
Tests for LambdaRank Tensorflow operation.

.. moduleauthor:: Nishant Sharma()
"""

import unittest
import numpy as np
import tensorflow as tf
from lambdarank import *
from testModel import Slide2VecScoringModel

def runTests():
    numSlides = 10
    numQueries = 5
    wordVecLenIn=300
    wordVecLenReduced=50
    slideVecLen=50
    input_length=1
    maxRating = numSlides/2

    scoringModel = Slide2VecScoringModel(
        wordVecLenIn=wordVecLenIn,
        wordVecLenReduced=wordVecLenReduced,
        slideVecLen=slideVecLen,
        slideCount=numSlides,
        input_length=input_length)
    X = np.random.rand(numSlides * numQueries, wordVecLenIn)
    y = np.random.randint(1, maxRating+1, numSlides * numQueries)
    rids = np.array([int(i%numSlides) for i in range(numSlides * numQueries)])
    qids =  [int(i/numSlides) for i in range(numSlides * numQueries)]

    # lambdaRankModel._calc_cross_lambdas(1.0, y, y_pred, debugIndex=-1, full=True, debug=True)
    scoringModel.fit(X, y, rids, qids, max_k=7)
    scoringModel.predict(X, rids)
    # print("Profiling data for fitting result:\n {0}".format(json.dumps(lastCallProfile(), indent=4)))


if __name__ == "__main__":
    runTests()
