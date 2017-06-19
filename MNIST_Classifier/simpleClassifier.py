from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# input data
x = tf.placeholder(tf.float32, [None, 784], name="input_X")

# weights and bias
W = tf.Variable(tf.zeros([784, 10]), name="weights")
b = tf.Variable(tf.zeros([10]), name="bias")

# output model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# training

# labeled output
y_ = tf.placeholder(tf.float32, [None, 10], name="labled_op")

# cross entropy loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# training with gradient descent optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# get session handle
sess = tf.InteractiveSession()

# init variables
tf.global_variables_initializer().run()

#logging
writer = tf.summary.FileWriter("./logs/", sess.graph)
tf.summary.scalar("loss", cross_entropy)
merged = tf.summary.merge_all()

# training 
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train, loss = sess.run([train_step, merged], feed_dict={x: batch_xs, y_: batch_ys})
  writer.add_summary(loss, i)

writer.close();

# prediction vs reality
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()
