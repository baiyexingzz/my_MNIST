import tensorflow as tf
import numpy as np
import csv
import math
import random

def generateVec(row, trainMode = True):

    pixel_vector = np.zeros(shape=(1, 784), dtype=np.float32)#[int(x) for x in row[1:]]

    if trainMode:
        label_vector = np.zeros(shape=(1, 10), dtype=np.float32)
        label_vector[0][int(row[0])] = 1
        for i in range(len(row)-1):
            pixel_vector[0][i] = int(row[i+1])
        return pixel_vector,label_vector
    else:
        for i in range(len(row)):
            pixel_vector[0][i] = int(row[i])
        return pixel_vector


def generate_batch(batch_size, trainFile, trainMode = True):
    pixels = np.ndarray(shape = (0,784), dtype = np.float32)
    if trainMode:
        labels = np.ndarray(shape = (0,10), dtype = np.float32)

    # trainFile = open('train.csv')
    reader = csv.reader(trainFile)
    trainNum = -1
    for row in reader:
        trainNum += 1
        if trainMode:
            pixel_vector, label_vector = generateVec(row)
            labels = np.vstack((labels, label_vector))
        else:
            pixel_vector = generateVec(row, trainMode = False)
        pixels = np.vstack((pixels, pixel_vector))
        if trainNum >= batch_size-1:
            break
    if trainMode:
        return pixels, labels
    else:
        return pixels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
    batch_size = 100
    num_train = 42000
    epochs_to_train = 15
    steps_to_train = math.floor(num_train // batch_size)

    x = tf.placeholder(dtype = tf.float32, shape=[None, 784], name='input')
    y_ = tf.placeholder(dtype = tf.float32, shape=[None, 10], name='output')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(tf.mul(y_,tf.log(y_conv)))
    # cross_entropy = tf.reduce_mean(tf.square(tf.sub(y_conv,y_)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # sess.run(tf.initialize_all_variables())

    # with tf.InteractiveSession()  as sess:
    sess=tf.Session()
    sess.run(tf.initialize_all_variables())
    #return

    print ('Start Train')
    for epoch in range(epochs_to_train):
        trainFileReader = open('train.csv', 'r+')
        print('epoch: ', epoch + 1)
        for step in range(int(steps_to_train)):
            pixels_batch, labels_batch = generate_batch(batch_size, trainFileReader)
            # print(labels_batch[0])
            feed_dict = {x : pixels_batch, y_ : labels_batch, keep_prob : 0.6}
            # print(sess.run(y_conv,feed_dict=feed_dict))
            # return
            _, loss_val,accuracy_val,temp = sess.run([train_step, cross_entropy,accuracy,b_conv1], feed_dict=feed_dict)

            if step % 10 == 9:
                # print ('Training at step ', step + 1, ' average loss: ', average_loss, ' accuracy: ', accuracy,'result: ',result)
                print ('Training at step ', step + 1, ' average loss: ', loss_val ,'result: ',accuracy_val)
        trainFileReader.close()

    print ('Start Test')
    test_file = open('test.csv','r')
    result_file = open('result.csv','w')
    result_writer = csv.writer(result_file)
    result_writer.writerow(["ImageId","Label"])
    batch_size = 1000
    test_num = 28000
    steps_to_test = math.floor(test_num // batch_size)
    cnt = 0
    for step in range(steps_to_test):
        print(step)

        pixels_batch = generate_batch(batch_size, test_file, trainMode=False)
        feed_dict = {x : pixels_batch, keep_prob : 1.0}
        predict = sess.run(tf.argmax(y_conv,1), feed_dict=feed_dict)

        for i in range(batch_size):
            #print fitness_val[i]
            cnt += 1
            result_writer.writerow([cnt, predict[i]])

    test_file.close()
    result_file.close()



if __name__ == '__main__':
    main()
