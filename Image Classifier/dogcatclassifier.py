import os
import numpy as np
import tensorflow as tf

import reader
import model

CLASSES = ['cat', 'dog']
N_CLASSES = len(CLASSES)
IMG_W = 208
IMG_H = 208
RATIO = 0.2
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 10000
learning_rate = 0.0003

reader = reader.Reader('./data/train/', CLASSES)
reader.split_image_list(RATIO)

train_img_batch, train_label_batch = reader.get_train_batch(IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
val_img, val_label = reader.get_val_batch(IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3], name="image")
y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="label")

logits = model.get_logits(x, BATCH_SIZE, N_CLASSES)
loss = model.get_loss(logits, y_)
accuracy = model.compare_prediction(logits, y_)
trainer = model.trainer(loss, learning_rate)

with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        
        summary_op = tf.summary.merge_all()        
        train_writer = tf.summary.FileWriter('./logs/train/', sess.graph)
        val_writer = tf.summary.FileWriter('./logs/val/', sess.graph)

	try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                
                tra_images,tra_labels = sess.run([train_img_batch, train_label_batch])
		print(step,'/',MAX_STEP)
                _, tra_loss, tra_acc = sess.run([trainer, loss, accuracy],feed_dict={x:tra_images, y_:tra_labels})
                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    summary_str = sess.run(summary_op, feed_dict={x:tra_images, y_:tra_labels})
                    train_writer.add_summary(summary_str, step)
                    
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_img, val_label])
                    val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x:val_images, y_:val_labels})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
                    summary_str = sess.run(summary_op, feed_dict={x:tra_images, y_:tra_labels})
                    val_writer.add_summary(summary_str, step)  
                                    
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join('./logs/train', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)

