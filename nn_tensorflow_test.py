import tensorflow as tf
import numpy as np
import csv
import math
import random

def generateVec(row):
    pixel_vector = np.zeros(shape=(1, 784), dtype=np.int32)#[int(x) for x in row[1:]]

    for i in range(len(row)):
        pixel_vector[0][i] = int(row[i])

    # print(row,pixel_vector)
    return pixel_vector

def generate_batch(batch_size, trainFile):
    pixels = np.ndarray(shape = (0,784), dtype = np.int32)

    # trainFile = open('train.csv')
    reader = csv.reader(trainFile)
    trainNum = -1
    for row in reader:
        trainNum += 1


        pixel_vector = generateVec(row)
        pixels = np.vstack((pixels, pixel_vector))
        if trainNum >= batch_size-1:
            # print(trainNum)
            break

    # print(np.shape(pixels), np.shape(labels),trainNum)
    return pixels


def add_layer(inputs, in_size, out_size, keep_prob, layer_name=None, activation_function=None):
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.random_uniform([in_size, out_size], minval=-math.sqrt(6.0/(in_size + out_size)), maxval=math.sqrt(6.0/(in_size + out_size))), name='Weights')
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biases')
        Wx_plus_b = tf.nn.dropout(tf.add(tf.matmul(inputs, Weights), biases), keep_prob=keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
    return outputs





def main():

    batch_size = 1000
    batch_index = 1
    num_test = 28000

    steps_to_test = num_test // batch_size

    result = [0,0]

    input_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 784], name='input')
    output_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 10], name='output')
    keep_prob_placeholder = tf.placeholder(tf.float32)
    global_step = tf.Variable(1, trainable=False, name='global_step')

    hidden_layer1 = add_layer(input_placeholder, 784, 1000, keep_prob_placeholder, activation_function=tf.nn.tanh , layer_name='hidden_layer')
    # hidden_layer2 = add_layer(hidden_layer1, 1000, 1000, keep_prob_placeholder, activation_function=tf.nn.tanh , layer_name='hidden_layer')
    output_layer = add_layer(hidden_layer1, 1000, 10, keep_prob_placeholder, activation_function=tf.nn.softmax  , layer_name='output_layer')

    saver = tf.train.Saver()
    test_file= open('test.csv', 'r')
    csv_file = open('result.csv','w')
    result_writer = csv.writer(csv_file)
    result_writer.writerow(["ImageId","Label"])
    index_cnt = 0

    with tf.Session() as sess:
        saver.restore(sess,'model.ckpt')
        total_correct = 0
        print ('Start test ...')

        for step in range(steps_to_test):
            print(step)

            pixels_batch = generate_batch(batch_size, test_file)
            feed_dict = {input_placeholder : pixels_batch, keep_prob_placeholder : 1.0}
            output = sess.run(output_layer, feed_dict=feed_dict)

            for i in range(batch_size):
                #print fitness_val[i]
                index_cnt += 1
                predict = 0
                for j in range(10):
                    if output[i][j] > output[i][predict]:
                        predict = j
                result_writer.writerow([index_cnt, predict])

        test_file.close()
        csv_file.close()

if __name__ == '__main__':
    main()
