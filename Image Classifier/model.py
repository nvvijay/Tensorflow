import tensorflow as tf


#```
#*--------------------------------------------------------------------*
#*                              MODEL SHAPE                           *
#*--------------------------------------------------------------------*
#*               image --> conv2d --> pool/norm -- ..                 *
#*                  .. --> conv2d --> pool/norm -- ..                 *
#*                  .. --> fc --> relu -- ..                          *
#*                  .. --> fc --> relu -- ..                          *
#*                  .. --> softmax --> class prediction               *
#*--------------------------------------------------------------------*
#
#
#```

# Defining model components

def make_conv_layer(input, w_shape, b_shape, stride, name):
	weights = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1, mean=0.0))
	biases = tf.Variable(tf.constant(0.1, shape=b_shape))

	conv2d = tf.nn.conv2d(input, weights, strides=stride, padding='SAME')
	with_bias = tf.nn.bias_add(conv2d, biases)
	conv_layer = tf.nn.relu(with_bias, name='conv_layer_'+name)

	return conv_layer

def pool_and_norm_layer(input, ksize, stride, name):
	pool = tf.nn.max_pool(input, ksize=ksize, strides=stride, padding='SAME', name='pool_and_norm_'+name)
	norm = tf.nn.local_response_normalization(pool, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='pool_and_norm_'+name)

	return norm

def norm_and_pool_layer(input, ksize, stride, name):
	norm = tf.nn.local_response_normalization(input, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='pool_and_norm_'+name)
	pool = tf.nn.max_pool(norm, ksize=ksize, strides=stride, padding='SAME', name='pool_and_norm_'+name)

	return pool

def make_fc_layer(input, w_shape, b_shape, name):
	weights = tf.Variable(tf.truncated_normal(w_shape, stddev=0.005, mean=0.0))
	biases = tf.Variable(tf.constant(0.1, shape=b_shape))

	fc_layer = tf.nn.relu(tf.matmul(input, weights) + biases, name='fc_layer_'+name)
	return fc_layer

def softmax_layer(input, w_shape, b_shape, name):
	weights = tf.Variable(tf.truncated_normal(w_shape, stddev=0.005, mean=0.0))
        biases = tf.Variable(tf.constant(0.1, shape=b_shape))

	softmax_layer = tf.add(tf.matmul(input, weights), biases, name='softmax_'+name)
	return softmax_layer
	

# Defining the model
def get_logits(images, batch_size, n_classes):
	initial_wshape = [3, 3, 3, 16]
	initial_bshape = [16]
	initial_stride = [1, 1, 1, 1]
	
	k_size = [1, 3, 3, 1]
	k_stride = [1, 2, 2, 1]

	conv1 = make_conv_layer(images, initial_wshape, initial_bshape, initial_stride, 'cl1')
	norm1 = pool_and_norm_layer(conv1, k_size, k_stride, 'pn1')
	
	w_shape = [3, 3, 16, 16]
	conv2 = make_conv_layer(norm1, w_shape, initial_bshape, initial_stride, 'cl2')
	pool1 = norm_and_pool_layer(conv2, k_size, initial_stride, 'np1')

	reshaped_layer = tf.reshape(pool1, shape=[batch_size, -1])
	new_shape = reshaped_layer.get_shape()[1].value
	
	fc1 = make_fc_layer(reshaped_layer, [new_shape, 128], [128], 'fc1')
	fc2 = make_fc_layer(fc1, [128, 128], [128], 'fc2')

	softmax = softmax_layer(fc2, [128, n_classes], [n_classes], 'sm1')
	return softmax

# Defining operations

def get_loss(logits, labels):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
	loss = tf.reduce_mean(cross_entropy, name='loss')
	tf.summary.scalar('loss', loss)

	return loss

def compare_prediction(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	correct = tf.cast(correct, tf.float16)
	accuracy = tf.reduce_mean(correct)

	tf.summary.scalar('accuracy', accuracy)
	return accuracy

def trainer(loss, learning_rate):
	optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        trainer = optimizer.minimize(loss, global_step= global_step)

	return trainer
