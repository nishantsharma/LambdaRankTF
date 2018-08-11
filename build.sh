TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
g++ -std=c++11 -shared lambdarank.cc -o liblambdarank.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
