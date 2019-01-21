This module implements LambdaRank as a tensorflow OP in C++. As an example application,
we use this OP as a loss function in our keras based deep ranking/recommendation engine.
The ranking application embeds slide objects into d-dimensional space(slide2vec), such
that we obtain best LambdaRank scores.

Building the OP module:
    Following commands were issued to build the module.

TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

g++ -std=c++11 -shared lambdarank.cc -o liblambdarank.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2


NOTES:
	Make sure that you always use batch_size as same as sample size. The underlying reason for that is
	all rows belonging to the same query must go into LambdaRankOp together.
	Example, when using TensorBoard with my keras model, I had to set the batch_size as INFINITY, because
	otherwise, I was risking incorrect computation.
