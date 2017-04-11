from featureExtraction import *
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from fileio import *

tf.reset_default_graph()

samplesize=4800
#samplesize=9635
# Network Parameters
complayersize=int(samplesize/1.3)
n_hidden_1 = int(samplesize/1) # 1st layer num features
n_hidden_2 = complayersize # 2nd layer num features
n_input = int(samplesize)
filename =os.getcwd()+  '/encoder/test_3.wav'
#filename =os.getcwd()+  '/encoder/b0366.compwav'

encode=True


ecnoderfile=os.getcwd()+"/encoder/modeltf.ckpt"

X = tf.placeholder("float", [None, n_input])
XC = tf.placeholder("float", [None, n_hidden_2])


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],mean=0.5)),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1],mean=0.5)),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
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



encodedY = encoder(X)
decodedY = decoder(XC)



samplesize=4800
init = tf.initialize_all_variables()

with tf.Session() as sess:
    saver = tf.train.Saver()

    sess.run(init)  # instead of init we gonna load the model

    saver.restore(sess, ecnoderfile)
    rawfile = []
    if (encode == True):
        rawfile, sample_rate = librosa.load(filename)
    else:
        f = open(filename, 'r')
        rawfile = np.fromfile(f,dtype=np.float32)

        f.close()
    pointer = 0
    keepprocessing = True
    sampletoprocess = []
    encodedsamples = []
    if(encode==False):
        samplesize=complayersize
    while keepprocessing:
        leftsample = len(rawfile) - pointer * samplesize


        if (leftsample < samplesize):
            sampletoprocess = rawfile[ pointer * samplesize:len(rawfile)]
            sampletoprocess= np.append(  sampletoprocess, [0] * (samplesize - leftsample))
            keepprocessing = False
        else:
            sampletoprocess = rawfile[pointer * samplesize:pointer * samplesize + samplesize]


        pointer=pointer+1
        if(encode==True):
            yarr=sess.run(encodedY,feed_dict={X: [sampletoprocess]})
            decodedYarr= sess.run(decodedY, feed_dict={XC: yarr})

            sqerror = tf.pow(sampletoprocess - decodedYarr, 2)
            cost = tf.reduce_mean(sqerror)
            print(cost.eval())

        else:
            yarr = sess.run(decodedY, feed_dict={XC: [sampletoprocess]})

        encodedsamples.append(yarr)

    filename = os.path.splitext(filename)[0]
    if (encode):

        filename = filename + ".compwav"
    else:

        filename = filename + "_b.wav"
    writebacktofile(filename, encodedsamples, encode)



