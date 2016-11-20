import tensorflow as tf
import numpy as np
import csv
import math
import random

def generateVec(row):
    label_vector = np.zeros(shape=(1, 10), dtype=np.int32)
    pixel_vector = np.zeros(shape=(1, 784), dtype=np.int32)#[int(x) for x in row[1:]]

    label_vector[0][int(row[0])] = 1
    for i in range(len(row)-1):
        pixel_vector[0][i] = int(row[i+1])

    # print(row,pixel_vector)
    return pixel_vector,label_vector

def generate_batch(batch_size, trainFile):
    pixels = np.ndarray(shape = (0,784), dtype = np.int32)
    labels = np.ndarray(shape = (0,10), dtype = np.int32)

    # trainFile = open('train.csv')
    reader = csv.reader(trainFile)
    trainNum = -1
    for row in reader:
        trainNum += 1

        pixel_vector, label_vector = generateVec(row)
        pixels = np.vstack((pixels, pixel_vector))
        labels = np.vstack((labels, label_vector))
        if trainNum >= batch_size-1:
            # print(trainNum)
            break

    # print(np.shape(pixels), np.shape(labels),trainNum)
    return pixels, labels


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
    num_train = 42000
    epochs_to_train = 15
    steps_to_train = math.floor(num_train // batch_size)
    learn_rate = 0.5

    input_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 784], name='input')
    labels_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 10], name='output')
    keep_prob_placeholder = tf.placeholder(tf.float32)
    global_step = tf.Variable(1, trainable=False, name='global_step')

    hidden_layer1 = add_layer(input_placeholder, 784, 1000, keep_prob_placeholder, activation_function=tf.nn.tanh , layer_name='hidden_layer')
    # hidden_layer2 = add_layer(hidden_layer1, 1000, 1000, keep_prob_placeholder, activation_function=tf.nn.tanh , layer_name='hidden_layer')
    output_layer = add_layer(hidden_layer1, 1000, 10, keep_prob_placeholder, activation_function=tf.nn.softmax , layer_name='output_layer')

    loss = tf.reduce_mean(tf.square(tf.sub(labels_placeholder,output_layer)))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    with tf.Session() as sess:
        sess.run(init)
        #return
        print ('Initialized')

        print ('Start Train')
        for epoch in range(epochs_to_train):
            average_loss = 0
            correct_num = 0
            result = [0,0]
            total_correct = 0
            trainFileReader = open('train.csv', 'r+')
            print('epoch: ', epoch + 1)
            for step in range(int(steps_to_train)):
                pixels_batch, labels_batch = generate_batch(batch_size, trainFileReader)
                if np.shape(labels_batch)[0]<200:
                    break
                while np.shape(labels_batch)[0]<1000:
                    rand_index = random.randint(0, np.shape(labels_batch)[0]-1)
                    pixels_batch = np.vstack((pixels_batch,pixels_batch[rand_index]))
                    labels_batch = np.vstack((labels_batch,labels_batch[rand_index]))

                feed_dict = {input_placeholder : pixels_batch, labels_placeholder : labels_batch, keep_prob_placeholder : 0.6}
                _, loss_val,output = sess.run([train, loss,output_layer], feed_dict=feed_dict)
                for i in range(batch_size):
                    #print fitness_val[i]
                    predict = 0
                    for j in range(10):
                        if output[i][j] > output[i][predict]:
                            predict = j
                    if labels_batch[i][predict] == 1:
                        result[0] += 1
                    else:
                        result[1] += 1
                correct_num = result[0]
                average_loss += loss_val
                if step % 10 == 9:
                    average_loss /= 10.0
                    accuracy = 1.0 * correct_num / (  batch_size*(step+1))
                    total_correct = correct_num
                    # print ('Training at step ', step + 1, ' average loss: ', average_loss, ' accuracy: ', accuracy,'result: ',result)
                    print ('Training at step ', step + 1, ' average loss: ', average_loss, ' accuracy: ', accuracy,'result: ',result)
                    correct_num = 0
                    average_loss = 0
                    # break
            trainFileReader.close()
            total_accuracy = total_correct / num_train
            print ('Train total accuracy :', total_accuracy)
            total_correct = 0
        save_path = saver.save(sess, 'model.ckpt')



if __name__ == "__main__":
    main()
