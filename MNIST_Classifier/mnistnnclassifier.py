import tensorflow as tf

#Loading MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#	i/p layer -> hl1(100) -> activation -> hl2(100) -> activation -> softmax -> output

#defining graph params
n_classes = 10
n_nodes_hl = [100, 100]
batch_size = 50

x = tf.placeholder(tf.float32, [None, 784], name='model_input')
y = tf.placeholder(tf.float32, name='prediction')

#define the model
def nn_model(data):
	hl1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl[0]]), 'weight at 1'), 'biases': tf.Variable(tf.random_normal([n_nodes_hl[0]]), 'bias at 1')}
	hl2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl[0], n_nodes_hl[1]]), 'weight at 2'), 'biases': tf.Variable(tf.random_normal([n_nodes_hl[1]]), 'bias at 2')}	
	opl = {'weights': tf.Variable(tf.random_normal([n_nodes_hl[1], n_classes]), 'weight at op'), 'biases': tf.Variable(tf.random_normal([n_classes]), 'bias at op')}
	
	l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
	l2 = tf.nn.relu(l2)

	op = tf.add(tf.matmul(l2, opl['weights']), opl['biases'])

	return op


#training the model
def train_model(data):
	model = nn_model(data)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#logging
        	writer = tf.summary.FileWriter("./logs/nnclassifier", sess.graph)
        	tf.summary.scalar("cost", cost)
		merged = tf.summary.merge_all()
		for epoch in range(20):
			total_loss = 0
			for i in range(int(mnist.train.num_examples/batch_size)):
				x_, y_ = mnist.train.next_batch(batch_size)
				_, loss, plot= sess.run([optimizer, cost, merged], feed_dict={x:x_, y:y_})
				total_loss += loss
				writer.add_summary(plot, epoch*int(mnist.train.num_examples/batch_size) + i)
			print(epoch,  '/20 : ', total_loss)
		
		correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))	
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_model(x)
