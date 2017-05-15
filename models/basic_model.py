# codes for blog
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

import numpy as np 

print('expected cross entroy loss if the model:')
print('- learns neither dependency: ', -(0.625*np.log(0.625) + 0.375*np.log(0.375)))

# learns first dependency only => 0.51916669970720941
print(' - learns first dependency: ', 
	-0.5*(0.875*np.log(0.875) + 0.125*np.log(0.125))
	-0.5*(0.625*np.log(0.625) + 0.375*np.log(0.375)))

print('-leatns both dependencies: ', 
	-0.50*(0.75*np.log(0.75) + 0.25*np.log(0.25))
	-0.25*(2*0.50*np.log(0.5))
	-0.25*(0))

# S_t = tanh(W(X_t@S_t−1)+b_s)
# P_t = softmax(U*S_t+b_p)

import tensorflow as tf 
# %matplotlib inline
# import matplotlib.pyplot as plt 

# global config variables
num_steps = 5
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1

x = tf.placeholder(tf.int32, [batch_size, num_steps], name = 'input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name = 'labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

'''
RNN Inputs
'''

x_one_hot = tf.one_hot(x, num_classes)
rnn_input = tf.unstack(x_one_hot, axis = 1)

'''
definition of rnn_cell
this is very similar to the __call__ method on tensorflow's BasicRNNCell, see
https://github.com/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L95
'''
with tf.variable_scope('rnn_cell'):
	W = tf.get_variable('W', [num_classes + state_size, state_size])
	b = tf.get_variable('b', [state_size], initializer = tf.constant_initializer(0.0))

def rnn_cell(rnn_input, state):
	with tf.variable_scope('rnn_cell', reuse = True):
		W = tf.get_variable('W', [num_classes + state_size, state_size])
		b = tf.get_variable('b', [state_size], initializer = tf.constant_initializer(0.0))
	return tf.tanh(tf.matmul(tf.concat([rnn_input, state]), W) + b)
	