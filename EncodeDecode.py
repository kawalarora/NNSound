
""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from featureExtraction import *



# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10
samplesize=4800
#samplesize=9635
# Network Parameters
n_hidden_1 = int(samplesize/1) # 1st layer num features
n_hidden_2 = int(samplesize/1.3) # 2nd layer num features
n_input = int(samplesize)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],mean=0.001)),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],mean=0.001)),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1],mean=0.001)),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input],mean=0.001)),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1],mean=0.001)),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2],mean=0.001)),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1],mean=0.001)),
    'decoder_b2': tf.Variable(tf.random_normal([n_input],mean=0.001)),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 =    tf.nn.sigmoid(  tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation  #2 tf.nn.sigmoid(
    layer_2 =  tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid( tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 =  tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2'])
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
sqerror=tf.pow(y_true - y_pred, 2)
cost = tf.reduce_mean(sqerror)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer (learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
ecnoderfile=os.getcwd()+"/encoder/modeltf.ckpt"
# Launch the graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(init)

    total_batch = 5
    outwav=[]
    srate=0
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for filename,samplerate,batch_xs in extract_wavonly(os.getcwd()+'/Dataset',10):

            batchn=100
            batch_xs=np.array(batch_xs)
            batch_xs=batch_xs.reshape(int(batchn),samplesize)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c  = sess.run([optimizer, cost ], feed_dict={X: batch_xs})
            if(c<0.1):
                saver.save(sess, ecnoderfile)
#                writebacktofile(filename)

        # Display logs per epoch step
      #  if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))


    print("Optimization Finished!")

