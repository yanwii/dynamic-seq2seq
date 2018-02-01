

import tensorflow as tf

char_inputs = [[2,1],[1,2],[2,3],[3,4],[4,0]]

used = tf.sign(tf.abs(char_inputs))
length = tf.reduce_sum(used, reduction_indices=0)
lengths = tf.cast(length, tf.int32)


sess = tf.Session()
print sess.run(lengths)
