import tensorflow as tf
import model
import reader
import numpy as np

num_classes = 2
debug = False

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

with tf.Session() as sess:

	saver = tf.train.import_meta_graph('./logs/train/model.ckpt-9999.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./logs/train/'))
	graph = tf.get_default_graph()
	image, label = graph.get_tensor_by_name("image:0"), graph.get_tensor_by_name("label:0")

	logits = graph.get_tensor_by_name("softmax_sm1:0")
	
	if debug:	
		ip = int(input("number :"))
		while ip > 0: 
			img_tensor = reader.get_image_as_tensor('./data/test/'+str(ip)+'.jpg', 208, 208)
			img = sess.run(img_tensor)

			feed_dict = {image:[img]*64}	
			prediction = sess.run(logits, feed_dict=feed_dict)
			prediction =  prediction.flatten().tolist()
			prediction = prediction[:num_classes]

			op = "cat" if prediction[0]>prediction[1] else "dog"
			print op, prediction
			ip = int(input("number :"))
def predict(img):
	feed_dict = {image:[img]*64}
	prediction = sess.run(logits, feed_dict=feed_dict)
        prediction =  prediction.flatten().tolist()
        prediction = prediction[:num_classes]

	return softmax(prediction)
