from featureExtraction import *
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from fileio import *
class mainTrainer:
    def __init__(self,n_classes,classtype):

        self.n_classes=n_classes

        file = fileio(classtype)
        self.parent_dir = file.TrainingFileFolder
        self.modelfile = file.TrainingFileFolder + "/lastRun/layernet/modeltf.ckpt"

    def setnetworktfparam(self):

        self.training_epochs = 100
        self.n_hidden_units_one = 500 #280
        self.n_hidden_units_two = 350 #300
        self.learning_rate = 0.01

    def dataprepsingle(self,filename):
        feature=getfeaturesforsingle(  filename)

        self.n_dim =  feature.shape[1]

        self.sd = 1 / np.sqrt(self.n_dim)

        return feature

    def dataprep(self):

        tr_sub_dirs = ['train']
        ts_sub_dirs = ['test']

        self.tr_features, self.tr_labels = parse_audio_files(self.parent_dir, tr_sub_dirs)
        self.ts_features, self.ts_labels = parse_audio_files(self.parent_dir, ts_sub_dirs)
        self.tr_labels = one_hot_encode(self.tr_labels)
        self.ts_labels = one_hot_encode(self.ts_labels)

        self.n_dim = self.tr_features.shape[1]

        self.sd = 1 / np.sqrt(self.n_dim)

    def networktf(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.n_classes])

        W_1 = tf.Variable(tf.random_normal([self.n_dim, self.n_hidden_units_one], mean=0, stddev=self.sd))
        b_1 = tf.Variable(tf.random_normal([self.n_hidden_units_one], mean=0, stddev=self.sd))
        h_1 = tf.nn.tanh(tf.matmul(self.X, W_1) + b_1)

        W_2 = tf.Variable(tf.random_normal([self.n_hidden_units_one, self.n_hidden_units_two], mean=0, stddev=self.sd))
        b_2 = tf.Variable(tf.random_normal([self.n_hidden_units_two], mean=0, stddev=self.sd))
        h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

        W = tf.Variable(tf.random_normal([self.n_hidden_units_two, self.n_classes], mean=0, stddev=self.sd))
        b = tf.Variable(tf.random_normal([self.n_classes], mean=0, stddev=self.sd))
        self.y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)



    def getcosttf(self):
        self.cost_function = -tf.reduce_sum(self.Y * tf.log(self.y_))
      #  self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_function)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost_function)
       # tf.train.AdagradOptimizer(self.learning_rate).minimize()
        correct_prediction = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def traintf(self):
        cost_history = np.empty(shape=[1], dtype=float)
        y_true, y_pred = None, None
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.training_epochs):
                _, cost = sess.run([self.optimizer, self.cost_function], feed_dict={self.X: self.tr_features, self.Y: self.tr_labels})
                cost_history = np.append(cost_history, cost)
                print(" epoc {0} , cost {1}".format(epoch,cost))

            y_pred = sess.run(tf.argmax(self.y_, 1), feed_dict={self.X: self.ts_features})
            y_true = sess.run(tf.argmax(self.ts_labels, 1))
            print('Test accuracy: ', round(sess .run(self.accuracy, feed_dict={self.X: self.ts_features, self.Y: self.ts_labels}), 3))
            saver.save(sess, self.modelfile)
        fig = plt.figure(figsize=(10, 8))
        plt.plot(cost_history)
        plt.axis([0, self.training_epochs, 0, np.max(cost_history)])
        plt.show()

        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
        print(
        "F-Score:", round(f, 3)
        )



    def classifytf(self,modelfile,singlefilefeature):
        init = tf.initialize_all_variables()

        saver = tf.train.Saver()
        sess = tf.Session()

        sess.run(init) #instead of init we gonna load the model
        saver.restore(sess, modelfile)
        y_pred = sess.run(tf.argmax(self.y_, 1), feed_dict={self.X: singlefilefeature})
        sess.close()
        return y_pred
    def classify(self,filename):

        feature= self.dataprepsingle(filename)
        tf.reset_default_graph()


        self.setnetworktfparam()

        self.networktf()
        y=self.classifytf(self.modelfile,feature)
        return y


    def trainermain(self):
        self.dataprep()
        self.setnetworktfparam()

        # all set to define tf network
        self.networktf()
        #now get cost
        self.getcosttf ()
        self.traintf()

def Train(n_classes , classtype):
     tr = mainTrainer(n_classes,classtype)
     tr.trainermain()
def Test(n_classes , classtype):
    tr = mainTrainer(n_classes, classtype)


    file = fileio(classtype)
# filename = file.TrainingFileFolder+ "/lastrun/Record_doggytraining_97_0.wav"
    filename = file.TrainingFileFolder + "/lastrun/Record_58.wav"

    a = tr.classify(filename)
    print("file {0} class {1}".format(filename, a))

if __name__ == '__main__':
   # n_classes=2
    #classtype="doggytraining"


    n_classes=5
    classtype="music_instrument"

# Train
    Train(n_classes,classtype)
# test
   # Test(n_classes, classtype)

