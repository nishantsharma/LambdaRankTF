This module implements LambdaRank as a tensorflow OP in C++. As an example application,
we use this OP as a loss function in our keras based deep learning ranking application.
The ranking application embeds slide objects into d-dimensional space(slide2vec), such
that we obtain best LambdaRank scores.

    At the time of this writing, there are other LambdaRank implementations available in open source. However, none of them can be integrated into TensorFlow to optimize
a deep learning model like this one. Test code is provided alongside.

Building the OP module:
    Following commands were issued to build the module.

TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
g++ -std=c++11 -shared lambdarank.cc -o liblambdarank.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
