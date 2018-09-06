import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv

'''
Here we globally define the start and the end
of train and test data.
'''

class Utils:
    # Add function to create weight variable
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # Add function to create bias variable
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Add convolution function
    def conv2d(x, W):
        return (tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME"))

    # Add Pooling function
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Root Mean Squared Error function
    def rmse(targets, outputs):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(targets, outputs))))

    def normalization(data):
        return data/255

    def batch(data, str, end, size):
        randIndexes = np.random.randint(str, end, size, int)
        return data[randIndexes]

class KeypointPredictor:
    def __init__(self, fileTrain, fileTest):
        '''
        Here we first of all read the
        training data file. Then we normalize the
        training data so as to hasten up the model
        training. Then test data file is read and normalized.
        :param fileTrain: Name of the training data file
        :param fileTest: Name of the testing data file
        '''
        self.epochs = 100
        self.xTrain, self.yTrain = self.read(fileTrain)
        self.xTrain = Utils.normalization(self.xTrain)
        self.xTrain, self.xValidation, self.yTrain, self.yValidation = train_test_split(self.xTrain, self.yTrain, test_size=0.75, random_state= 41)
        self.xTest = self.read(fileTest)
        self.xTest = Utils.normalization(self.xTest)
        self.pred = self.modelCNNPredictions()
        self.saveResults("Predictions.csv")


    def mapper(self, flag, row):
        '''
        This is a mapping utility
        for modifying a list to have
        float datatypes
        :param flag: this tells the purpose of mapping
        :param row: data row
        :return:
        '''
        try:
            if flag == "train X" :
                output = lambda data: [float(val) for val in data[30].split(' ')]
            elif flag == "train Y":
                output = lambda data: [float(val) if (val != '') else None for val in data[0:30]]
            elif flag == "test X":
                output = lambda data: [float(val) for val in data[1].split(' ')]
            return output(row)
        except :
            print("Mapper function not available")

    def read(self, fileName):
        '''
        Had a issue while reading
        with np.readcsv as it reads
        everything in memory
        :param fileName: Name of the file
        :return: read data
        '''
        try:
            with open(fileName, newline='') as myFile:
                reader = csv.reader(myFile)
                next(reader, None)
                if ("train" in fileName):
                    xTrain = []
                    yTrain = []
                    for row in reader:
                        xTrain.append(self.mapper("train X", row))
                        yTrain.append(self.mapper("train Y", row))
                    return np.array(xTrain), np.array(yTrain)
                elif ("test" in fileName):
                    xTest = []
                    for row in reader:
                        xTest.append(self.mapper("test X", row))
                    return np.array(xTest)
        except:
            print("File not present for Loading")

    def modelSklearn(self):
        '''
        There is no solution using
        sklearn as of now.
        '''
        model = sk.linear_model(self.xTrain, self.yTrain)
        predictions = model.predict(self.xValidation)
        # TODO: Use some optimization along with a loss function to reduce error
        return predictions

    def modelCNNPredictions(self):
        '''
        This is the tensorflow convolutional
        Neural Network model that we need to
        design so as to find features from
        images dataset. First of all we create
        computational Graph then we will just
        execute it.
        :return: a CNN model
        '''
        # These are the placeholders to hold the training X and Y data
        x = tf.placeholder(tf.float32, [None, 9216])
        y_ = tf.placeholder(tf.float32, [None, 30])

        # Reshape the data for inputting it in the input layer
        x_image = tf.reshape(x, [-1, 96, 96, 1])


        # Create 1st Convolutional and Pooling Layer
        W_conv1 = Utils.weight_variable([5, 5, 1, 32])
        b_conv1 = Utils.bias_variable([32])
        h_conv1 = tf.nn.relu(Utils.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = Utils.max_pool_2x2(h_conv1)


        # Create 2nd Convolutional and Pooling Layer
        W_conv2 = Utils.weight_variable([5, 5, 32, 64])
        b_conv2 = Utils.bias_variable([64])
        h_conv2 = tf.nn.relu(Utils.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = Utils.max_pool_2x2(h_conv2)


        # Create 3rd Convolutional and Pooling Layer
        W_conv3 = Utils.weight_variable([5, 5, 64, 128])
        b_conv3 = Utils.bias_variable([128])
        h_conv3 = tf.nn.relu(Utils.conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = Utils.max_pool_2x2(h_conv3)


        # 1st Fully Connected Layer
        W_fc1 = Utils.weight_variable([128 * 12 * 12, 1024])
        b_fc1 = Utils.bias_variable([1024])

        # Reshape the last convolutional layer output to connect with Fully Connected Layer
        h_pool2_flat = tf.reshape(h_pool3, [-1, 128 * 12 * 12])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


        # Add the dropout layer
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Output Layer
        W_fc2 = Utils.weight_variable([1024, 30])
        b_fc2 = Utils.bias_variable([30])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


        # Network base of Computation
        errorFunc = Utils.rmse(y_, y_conv)
        #errorFunc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, y_conv))))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(errorFunc)

        # Mention about the number of epochs training has to happen
        epochs = 1100

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                # Read a batch of training Data
                batchX = Utils.batch(self.xTrain, 0, 1761, 50)
                batchY = Utils.batch(self.yTrain, 0, 1761, 50)
                if(i % 100 == 0):
                    print("Beep Beep")
                train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})
            # Predict the output of the testing Data
            return y_conv.run(feed_dict={x: self.xTest, keep_prob: 0.5})


    def saveResults(self, fileName):
        with open(fileName) as myFile:
            myFile.write(self.pred)
            myFile.close()

if __name__ == '__main__':
    print(1)
    obj = KeypointPredictor("data/training.csv", "data/test.csv" )
    #print(len(obj.xTrain))
    #print(len(obj.yTrain))
    #print(len(obj.xTest))
    print("Done")