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

# Using float64. But float32 may work equally well.
K.set_floatx("float64")

# Activate tfdbg debugging.
# K.set_session(tf_debug.LocalCLIDebugWrapperSession(K.get_session()))

class Slide2VecRankingModel(object):
    """
    A LambdaRank model to search for slide objects.
    Only input for this model are the user ratings for keyword search results.
    It uses those ratings to embed slides into a d-dimensional space such that
    LambdaRank scores are optimal.
    """
    def __init__(self, queryVecLenIn, slideVecLen, slideCount, input_length):
        """
            queryVecLenIn: Length of each query word vector. Use of word2vec embedding is implied.
            slideVecLen: Length of slide vector embedding to find.
            slideCount: Number of slides in the search universe.
        """
        if input_length is None:
            input_length = slideCount
        self.queryVecLenIn = queryVecLenIn

        # Build input layers.
        query_vec_input = Input(shape=(queryVecLenIn,), name='queryVecInput')
        slide_index_input = Input(shape=(input_length,), dtype='int32', name='slideIndexInput')
        query_index_input = Input(shape=(input_length,), dtype='int32', name='queryIndexInput')

        # Build the slide embedding layer.
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
        # queryVecLen * slideVecLen
        # queryVecDimReducedAndAligned = MatrixProductLayer(slideVecLen, name="wordSlideAlignmentMatrix")(queryVecDimReduced)
        queryVecDimReducedAndAligned_constraint=constraints.UnitNorm(axis=(-2,))
        queryVecDimReducedAndAligned_initializer=random_initializer(axis=(-2,))
        queryVecDimReducedAndAligned = MatrixProductLayer(
            slideVecLen,
            weight_initializer=queryVecDimReducedAndAligned_initializer,
            weight_constraint=queryVecDimReducedAndAligned_constraint,
            name="wordSlideAlignmentMatrix")(query_vec_input)

        # Dot product of query word's vector and embedding vector of the slide.
        dotProductOutputs = Dot(-1)([slide_embedding, queryVecDimReducedAndAligned])

        # Our loss function needs to know about query ID, in addition to the query
        # vector. For that purpose, bundling query indices into the predicted output.
        query_index_floats = Lambda(lambda x:tf.to_double(x))(query_index_input)
        concatedOutput = Concatenate()([dotProductOutputs, query_index_floats])

        # Define the model.
        self.model = Model(
            inputs=[query_vec_input, query_index_input, slide_index_input],
            outputs=concatedOutput)

    @classmethod
    def __getIntQids(qids):
        """
        We need query IDs to be integers. Convert a non-int query ID vector to
        integer vectors, if they are not already.
        """
        if not isinstance(qids, np.ndarray) or qids.dtype != "int":
            qidMap = {}
            for qid in set(qids):
                qidMap[qid] = len(qidMap)
            intQids = np.ones([num_samples])
            for i in range(num_samples):
                intQids[i] = qidMap[qids[i]]
            return intQids

    def fit(self, X, y, qids, rids, max_k=20, epoch=200):
        """
        Function to fit search ratings data. Each training data row is indexed by
        (queryWord, slide).

        Important Note: Many (queryWord, slide) may share the same queryWord. But, all 
        rows wiht same queryWord must appear contiguous.

        Inputs:
            X: Vector generated from (queryWord, slide) pair.
            y: User ratings vector.
            qids: Query IDs. Can be integer or string vector.
            rids: Slide IDs. Must be integer vector.
        Output:
            Predicted scores for each data row after training.
        """
        qids = self.__getIntQids(qids)
        if not isinstance(rids, np.ndarray):
            rids = np.array(rids)

        # Loss calculations need to know max_k.
        lossObject = LambdaRankLoss(max_k=max_k)

        # Optimization object. Adam worked well for us.
        # optiObject = SGD(lr=0.001, decay=1e-5, momentum=0.4, nesterov=True)
        optiObject = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # Compile the model.
        self.model.compile(loss=lossObject, optimizer=optiObject)

        # Prepare inputs and send model.to fit.
        temp_y = y.reshape(y.shape[0],1).astype("float64")
        temp_qids = np.array(qids, dtype="float64").reshape(len(qids),1)
        y_to_send=np.concatenate([temp_y, temp_qids], axis=-1)
        x_to_send=[X, qids, rids]
        num_samples = len(X)

        ## IMPORTANT ##
        # DONT change batch_size. 
        # AND
        # DONT shuffle.
        ## /IMPORTANT ##
        plot_model(self.model, to_file="plot.png")
        self.model.fit(x_to_send, y_to_send, batch_size=num_samples, shuffle=False, epochs=epoch)

    def predict(self, X, qids, rids):
        # Prepare inputs and send to model.fit.
        qids = self.__getIntQids(qids)
        if not isinstance(rids, np.ndarray):
            rids = np.array(rids)
        num_samples = len(X)
        ## IMPORTANT ##
        # DONT change batch_size. 
        ## /IMPORTANT ##
        return self.model.predict([X, qids, rids], batch_size=num_samples)

def random_initializer(axis):
    """
    To initialize a tensor with random values such that:
    a) L2 Norm of each sub-tensor along input axes is 1.
    b) Matrix is sampled from normal distribution. That means that each individual
       sub-tensor is picked in a spherically symmetrical manner.
    """
    def func(shape, dtype=None):
        retval = np.random.normal(size=shape)
        retval /= np.sqrt(np.sum(retval*retval, axis=axis, keepdims=True))
        return retval
    return func

def random_init_unit_norm(shape, dtype=None):
    """
    To initialize a tensor with random values such that:
    a) L2 Norm for entire matrix is 1.
    b) Data is sampled from normal distribution. That means that each individual
       sub-tensor is picked in a spherically symmetrical manner.
    """
    retval = np.random.normal(size=shape, dtype=dtype)
    retval /= np.linalg.norm(retval)
    return retval

class MatrixProductLayer(Layer):
    """
    A keras layer to multiply the input matrix with a trainable weight kernel.
    """
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

