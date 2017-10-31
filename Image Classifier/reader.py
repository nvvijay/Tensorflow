import tensorflow as tf
import numpy as np
import os
import math 

class Reader:
	file_dir = ""
	class_list = []
	images_in_classes = {}
	labels_in_classes = {}

	train_images = []
	train_labels = []
	val_images = []
	val_lables = []

	def __init__(self, file_dir, class_list):
		self.file_dir = file_dir
		self.class_list = class_list
		self.images_in_classes = {x:[] for x in class_list}
		self.labels_in_classes = {x:[] for x in class_list}

	def split_image_list(self, ratio):
		for file in os.listdir(self.file_dir):
			name = file.split('.')[0]
			self.images_in_classes[name].append(self.file_dir + file)
			self.labels_in_classes[name].append(self.class_list.index(name))

		all_images = np.hstack(tuple(self.images_in_classes.values()))
		all_labels = np.hstack(tuple(self.labels_in_classes.values()))

		print("read "+str(all_images.size)+" images in total")

		temp = np.array([all_images, all_labels])
		temp = temp.transpose()
		np.random.shuffle(temp)

		all_image_list = temp[:, 0]
		all_label_list = temp[:, 1]
    
		n_sample = len(all_label_list)

		n_val = int(math.ceil(n_sample*ratio))
		n_train = int(n_sample - n_val)
   
		print("train count: "+str(n_train)+", validation count: "+str(n_val)) 
		self.train_images = all_image_list[0:n_train]
		self.train_labels = all_label_list[0:n_train]

		self.val_images = all_image_list[n_train:-1]
		self.val_labels = all_label_list[n_train:-1]

	def get_train_batch(self, img_w, img_h, batch_size, capacity):
		return self.get_batch(self.train_images, self.train_labels, img_w, img_h, batch_size, capacity)

	def get_val_batch(self, img_w, img_h, batch_size, capacity):
		return self.get_batch(val_images, val_labels, img_w, img_h, batch_size, capacity)
	
	def get_batch(self, images, labels, width, height, batch_size, capacity):
		images = tf.cast(images, tf.string)
		labels = tf.cast(labels, tf.int32)

		ip_q = tf.train.slice_input_producer([images, labels])
		labels = ip_q[1]
		images = tf.image.decode_jpeg(tf.read_file(ip_q[0]), channels=3)
		images = tf.image.resize_image_with_crop_or_pad(images, width, height)
		images = tf.image.per_image_standardization(images)
    
		image_batch, label_batch = tf.train.batch([images, labels], batch_size= batch_size, num_threads=64, capacity=capacity)
		label_batch = tf.reshape(label_batch, [batch_size])
		image_batch = tf.cast(image_batch, tf.float32)

		return image_batch, label_batch


r = Reader('./data/train/', ['cat', 'dog'])
r.split_image_list(0.2)
print(r.get_train_batch(208,208,64,256))


