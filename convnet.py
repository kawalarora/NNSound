from featureExtraction import *
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from fileio import *
class ConvNetTF:
    def __init__(self,n_classes,classtype):

        self.n_classes=n_classes

        file = fileio(classtype)
        self.parent_dir = file.TrainingFileFolder
        self.modelfile = file.TrainingFileFolder + "/lastRun/convnet/modeltf.ckpt"

        self.convsettings()


    def windows(self,data, window_size):
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size  )

    def extract_features(self,parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
        window_size = 512 * (frames - 1)
        log_specgrams = []
        labels = []
        for l, sub_dir in enumerate(sub_dirs):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                sound_clip, s = librosa.load(fn)
                label = fn [len(fn)-5]
                for (start, end) in self.windows(sound_clip, window_size):
                    if (len(sound_clip[int(start):int(end)]) == window_size):
                        signal = sound_clip[int(start):int(end)]
                        melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                        logspec = librosa.logamplitude(melspec)
                        logspec = logspec.T.flatten()[:, np.newaxis].T
                        log_specgrams.append(logspec)
                        labels.append(label)

        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

        return np.array(features), np.array(labels, dtype=np.int)

    def one_hot_encode(labels):
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), labels] = 1
        return one_hot_encode

    def convsettings(self):
        self.frames = 41
        self.bands = 60

        self.feature_size = 2460  # 60x41
        self.num_labels = self.n_classes
        self.num_channels = 2

        self.batch_size = 50
        self.kernel_size = 10
        self.depth = 30
        self.num_hidden = 100

        self.learning_rate = 0.01
        self.total_iterations = 20


    def convnet(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.bands, self.frames, self.num_channels])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_labels])

        cov = self.apply_convolution(self.X, self.kernel_size, self.num_channels, self.depth)
        cov = self.apply_max_pool(cov,self.kernel_size,4 )
        shape = cov.get_shape().as_list()
        cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

        f_weights = self.weight_variable([shape[1] * shape[2] * self.depth, self.num_hidden])
        f_biases = self.bias_variable([self.num_hidden])
        f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights), f_biases))

        out_weights = self.weight_variable([self.num_hidden, self.num_labels])
        out_biases = self.bias_variable([self.num_labels])
        self.y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(1.0, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

    def loss(self):
        self.loss = -tf.reduce_sum(self.Y * tf.log(self.y_))
        self.optimizer = tf.train.AdadeltaOptimizer (learning_rate=self.learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def apply_convolution(self,x,kernel_size,num_channels,depth):
        weights = self.weight_variable([kernel_size, kernel_size, num_channels, depth])
        biases = self.bias_variable([depth])
        return tf.nn.relu(tf.add(self.conv2d(x, weights),biases))

    def apply_max_pool(self,x,kernel_size,stride_size):
        return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                              strides=[1, stride_size, stride_size, 1], padding='SAME')

    def dataprep(self):

        tr_sub_dirs = ['train']
        ts_sub_dirs = ['test']

        self.tr_features, tr_labels = self.extract_features(self.parent_dir, tr_sub_dirs)
        self.tr_labels = one_hot_encode(tr_labels)

        self.ts_features, ts_labels = self.extract_features(self.parent_dir, ts_sub_dirs)
        self.ts_labels = one_hot_encode(ts_labels)

    def RunSession(self):
        cost_history = np.empty(shape=[1], dtype=float)
        with tf.Session() as session:
            tf.initialize_all_variables().run()

            for itr in range(self.total_iterations):
                offset = (itr * self.batch_size) % (self.tr_labels.shape[0] - self.batch_size)
                batch_x = self.tr_features[offset:(offset + self.batch_size), :, :, :]
                batch_y = self.tr_labels[offset:(offset + self.batch_size), :]

                _, c = session.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})
                cost_history = np.append(cost_history, c)
                print(c)
            print('Test accuracy: ', round(session.run(self.accuracy, feed_dict={self.X: self.ts_features, self.Y: self.ts_labels}), 3))
            fig = plt.figure(figsize=(15, 10))
            plt.plot(cost_history)
            plt.axis([0, self.total_iterations, 0, np.max(cost_history)])
            plt.show()


    def Train(self):
        self.dataprep()
        self.convnet()
        self.loss()
        self.RunSession()

if __name__ == '__main__':
    convent=ConvNetTF(5,"music_instrument")
    convent.Train()

