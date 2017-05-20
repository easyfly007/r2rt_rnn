# codes for blog
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

import numpy as np 

print('expected cross entroy loss if the model:')
print('- learns neither dependency: ', -(0.625*np.log(0.625) + 0.375*np.log(0.375)))

# if we cannot find the dependecncy, then it will only obbserved that the 1 probability is 
# (0.5 + 0.25 + 0.75 + 1) / 4 = 0.625, and the 0 probability is 0.375 

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

'''
adding rnn_cells to graph
this is a simplified version of the static_rnn function from tensorflow's api, see
https://github.com/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn.py#L41
Note: in practice, using dynamic_rnn is a better choice that the static_rnn:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py#L390
'''

state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
	state = rnn_cell(rnn_input, state)
	rnn_outputs.append(state)
final_state = rnn_outputs[-1]

'''
predictions, loss, training step
losses is similar to the sequence_loss
function from tensorflow's api, except that here we are using a list of 2D tensors, instead of a 3D tensor.see
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/loss.py
'''

#logits and preedictions
with tf.variable_scope('softmax'):
	w = tf.get_variable('W', [state_size, num_classes])
	b = tf.get_variable('b', [num_classes], initializer = tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_outputs, W) + b for rnn_outputs in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num = num_steps, axis = 1)

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits = logit) for
logit, label in zip(logits, y_as_list) ]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


def train_network(num_epochs, num_steps, state_size = 4, verbose = True):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		training_losses = []
		for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
			training_loss = 0
			training_state = np.zeros((batch_size, state_size))
			if verbose:
				print('\nEPOCH', idx)
			for step, (X, Y) in enumerate(epoch):
				# tr_losses, training_loss_, training_state, _ =
				sess.run([losses, total_loss, final_state, train_step], 
					feed_dict = {x:X, y:Y, init_state:training_state})
				training_loss += training_loss_
				if step % 100 == 0 and step >0:
					if verbose:
						print('average loss at step', step, 'for last 250 steps:', training_loss/100)
					training_losses.apend(training_loss / 100)
					training_loss = 0
	return training_losses

training_losses = train_network(1, num_steps)
# plt.plot(training_losses)

