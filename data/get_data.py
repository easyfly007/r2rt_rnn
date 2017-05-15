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
import matplotlib.pyplot as plt 

# global config variables
num_steps = 5
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1

def gen_data(size = 1000000):
	X = np.array(np.random.choice(2, size = (size, )))
	Y = []
	for i in range(size):
		threshold = 0.5
		if X[i-3] == 1:
			threshold += 0.5
		if X[i-8] == 1:
			threshold -= 0.25
		if np.random.rand() > threshold:
			Y.append(0)
		else:
			Y.append(1)
	return X, np.array(Y)

def gen_batch(raw_data, batch_size, num_steps):
	raw_x, raw_y = raw_data
	data_length = len(raw_x)

	batch_partion_length = data_length // batch_size
	data_x = np.zeros([batch_size, batch_partion_length], dtype = np.int32)
	data_y = np.zeros([batch_size, batch_partion_length], dtype = np.int32)

	for i in range(batch_size):
		data_x[i] = raw_x[batch_partion_length*i:batch_partion_length*(i+1)]
		data_y[i] = raw_y[batch_partion_length*i:batch_partion_length*(i+1)]
	epoch_size = batch_partion_length // num_steps

	for i in range(epoch_size):
		x = data_x[:, i*num_steps:(i+1)*num_steps]
		y = data_y[:, i*num_steps:(i+1)*num_steps]
		yield (x, y)

def gen_epochs(n, num_steps):
	for i in range(n):
		yield gen_batch(gen_data(), batch_size, num_steps)
		