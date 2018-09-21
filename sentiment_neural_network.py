import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

from create_sentiments_featuresets import create_feature_sets_and_labels

train_x,train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

#nodes for the hidden layers(it can be any number you want.)
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10 

#this will make batches of the data and feed them one at a time.
batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])  #([height, width(28*28)])
y = tf.placeholder('float')

def neural_network_model(data):

    #input data * weights + biases

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases' : tf.Variable(tf.random_normal([n_classes]))}
    

    layer1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    layer1 = tf.nn.relu(layer1) # passing the value to activation function.

    layer2 = tf.add(tf.matmul(layer1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer2 = tf.nn.relu(layer2) # passing the value to activation function.

    layer3 = tf.add(tf.matmul(layer2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer3 = tf.nn.relu(layer3) # passing the value to activation function.

    output = tf.matmul(layer3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels=y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost) #learning rate for AdamOptimizer=0.001

    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss +=c
                i += batch_size
            print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x,y:test_y}))

train_neural_network(x)