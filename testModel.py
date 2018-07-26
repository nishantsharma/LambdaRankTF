import urllib, json
import collections
import os
import zipfile

from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge, Dot, Lambda, Concatenate 
from keras.layers.embeddings import Embedding
from keras.layers import constraints
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import SGD, Adam
from keras.preprocessing.text import one_hot
from tensorflow.python.framework.ops import IndexedSlicesValue
from keras.utils import plot_model

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import sklearn.preprocessing
from lambdarank import *

K.set_floatx("float64")

# Activate tfdbg debugging.
# K.set_session(tf_debug.LocalCLIDebugWrapperSession(K.get_session()))

def random_initializer(axis):
    def func(shape, dtype=None):
        retval = np.random.normal(size=shape)
        retval /= np.sqrt(np.sum(retval*retval, axis=axis, keepdims=True))
        return retval
    return func

def random_init_unit_norm(shape, dtype=None):
    retval = np.random.normal(size=shape, dtype=dtype)
    retval /= np.linalg.norm(retval)
    return retval

class MatrixProductLayer(Layer):
    def __init__(self,
                 output_dim,
                 weight_constraint=None,
                 weight_initializer=random_init_unit_norm,
                 **kwargs):
        self.output_dim = output_dim
        self.weight_constraint = weight_constraint
        self.weight_initializer = weight_initializer
        super(MatrixProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      trainable=True,
                                      initializer=self.weight_initializer,
                                      constraint=self.weight_constraint)
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class Slide2VecScoringModel(object):
    def __init__(self, wordVecLenIn, wordVecLenReduced, slideVecLen, slideCount, input_length):
        if input_length is None:
            input_length = slideCount
        self.wordVecLenIn = wordVecLenIn
        wordvec_input = Input(shape=(wordVecLenIn,), name='queryVecInput')

        slide_index_input = Input(shape=(input_length,), dtype='int32', name='slideIndexInput')

        query_index_input = Input(shape=(input_length,), dtype='int32', name='queryIndexInput')

        # slideCount * slideVecLen
        slide_embeddings_constraint=constraints.UnitNorm(axis=(-1,))
        slide_embeddings_initializer=random_initializer(axis=(-1,))
        slide_embedding = Embedding(slideCount,
                                    slideVecLen,
                                    input_length=input_length,
                                    name='slideEmbedding',
                                    embeddings_initializer=slide_embeddings_initializer,
                                    embeddings_constraint=slide_embeddings_constraint
                                    )(slide_index_input)

        # Rotating/aligning dimensionally reduced word2vec vectors before taking dot product with slide vectors.
        # wordVecLenReduced * slideVecLen
        # wordVecDimReducedAndAligned = MatrixProductLayer(slideVecLen, name="wordSlideAlignmentMatrix")(wordvecDimReduced)
        wordVecDimReducedAndAligned_constraint=constraints.UnitNorm(axis=(-2,))
        wordVecDimReducedAndAligned_initializer=random_initializer(axis=(-2,))
        wordVecDimReducedAndAligned = MatrixProductLayer(
            slideVecLen,
            weight_initializer=wordVecDimReducedAndAligned_initializer,
            weight_constraint=wordVecDimReducedAndAligned_constraint,
            name="wordSlideAlignmentMatrix")(wordvec_input)

        dotProductOutputs = Dot(-1)([slide_embedding, wordVecDimReducedAndAligned])

        query_index_floats = Lambda(lambda x:tf.to_double(x))(query_index_input)

        concatedOutput = Concatenate()([dotProductOutputs, query_index_floats])
        self.model = Model(
            inputs=[wordvec_input, slide_index_input, query_index_input],
            outputs=concatedOutput)

    def fit(self, X, y, rids, qids, max_k=20):
        if not isinstance(qids, np.ndarray):
            qids = np.array(qids)
        if not isinstance(rids, np.ndarray):
            rids = np.array(rids)
        # Loss calculatoin needs to know result Ids and query Ids.
        lossObject = LambdaRankLoss(max_k=max_k)
        optiObject = SGD(lr=0.001, decay=1e-5, momentum=0.4, nesterov=True)
        # optiObject = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss=lossObject, optimizer=optiObject)
        # self.model.compile(loss="mean_squared_error", optimizer=optiObject)

        # Prepare inputs and send to model.fit.
        temp_y = y.reshape(y.shape[0],1).astype("float64")
        temp_qids = np.array(qids, dtype="float64").reshape(len(qids),1)
        y_to_send=np.concatenate([temp_y, temp_qids], axis=-1)
        x_to_send=[X, rids, qids]
        num_samples = X.shape[0]

        ## IMPORTANT ##
        # DONT change batch_size. 
        # AND
        # DONT shuffle.
        ## IMPORTANT ##
        plot_model(self.model, to_file="plot.png")
        self.model.fit(x_to_send, y_to_send, batch_size=num_samples, shuffle=False, epochs=10000)

    def predict(self, X, rids, qids):
        # Prepare inputs and send to model.fit.
        ## IMPORTANT ##
        # DONT change batch_size. 
        ## IMPORTANT ##
        num_samples = X.shape[0]
        return self.model.predict([X, rids, qids], batch_size=num_samples)

    def get_weights(self):
        retval = K.get_session().run(self.model.trainable_weights)
        return retval

    def set_weights(self, weights):
        opsList = []
        for (var, value) in zip(self.model.trainable_weights, weights):
            opsList.append(var.assign(value))
        return K.get_session().run(opsList)

    def init(self):
        sess = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, "kernel"):
                layer.kernel.initializer.run(session=sess)

        weights = self.get_weights()
        self.printNorms(weights)
        self.allWeights = [weights]

    def printNorms(self, weights):
        slideEmbedding = weights[0]
        slideEmbeddingNorm = np.sqrt(np.sum(
            slideEmbedding * slideEmbedding,
            axis=-1))
        print("\nSlide embedding norm shape={0}, diff-from-1={1}.".format( #, norm=={2}
            slideEmbeddingNorm.shape,
            np.sum(np.abs(slideEmbeddingNorm-1)),
            slideEmbeddingNorm,
            ))

        wordSlideAlignmentMatrix = weights[1]
        wordSlideAlignmentMatrixNorm = np.sqrt(np.sum(
            wordSlideAlignmentMatrix * wordSlideAlignmentMatrix,
            axis=-2))
        print("\nAlignment matrix norm shape={0}, diff-from-1={1}.\n".format( #, norm=={2}
            wordSlideAlignmentMatrixNorm.shape,
            np.sum(np.abs(wordSlideAlignmentMatrixNorm-1)),
            wordSlideAlignmentMatrixNorm,
            ))
