from keras.layers import *
from keras.activations import softmax
import tensorflow as tf

########
tensor = tf.convert_to_tensor([
          [2.0, 1.0, 0.1],
          [0.1, 1.0, 2.0],
          [2.0, 0.1, 1.0],
            [2.0, 0.1, 1.0]
        ])

with tf.Session() as sess:
    t = softmax(tensor)
    ris = sess.run(t)
    print(ris)

print(tensor.get_shape().as_list()[-1])