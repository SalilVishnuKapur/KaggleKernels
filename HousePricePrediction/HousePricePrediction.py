import sklearn as sk
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


class FeatureEngineering:
    def columnDetector(row):
        '''
        This detects which all
        columns are not integers
        and instead have some string
        stuff into it.
        :return: List of column numbers which are not strings
        '''
        try:
            lt = []
            for token, itr in zip(row, range(len(row))):
                try:
                    float(token)
                    lt.append(itr)
                except:
                    pass
            return lt
        except:
            print("This is an issue with the Detection")


    def columnRemoval(data):
        '''
        Only keep the columns from
        the data which are important
        for building up the model
        remove the rest of them
        Remove =>
        [0, 1, 3, 4, 17, 18, 19, 20, 26, 34, 36, 37, 38, 43,
         44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56, 59, 61,
         62, 66, 67, 68, 69, 70, 71, 75, 76, 77, 80]
        :return: Filtered piece of data
        '''
        return data[:, [0, 1, 3, 4, 17, 18, 19, 20, 26, 34, 36, 37, 38, 43, 44, 45, 46, 47, 48, 49, 50,
                        51, 52, 54, 56, 59, 61, 62, 66, 67, 68, 69, 70, 71, 75, 76, 77]]


    def typesLabeller(coldata):
        '''
        This method is quite generic
        which works on labelling the
        different classified things
        into their respective groups.
        :return: Numerically Labelled Column values
        '''
        pass



class HousePricePrediction:
    def __init__(self, flag, fileNameTrain, fileNameTest):

        # 1. Read the Data
        self.dataTrainX, self.dataTrainY = self.read(fileNameTrain)
        self.dataTestX = self.read(fileNameTest)
        print("Data Reading Successful")

        if(flag == True):
            # 1. Remove the unwanted columns
            self.dataTrainX = FeatureEngineering.columnRemoval(self.dataTrainX)
            self.dataTestX = FeatureEngineering.columnRemoval(self.dataTestX)
            print("Removing unwanted data Successful")

            # 2. Apply the mathematical model to it
            self.model = self.mlModel()
            print("Linear Regression Model is ready")

            # 3. Get the predictions
            self.predictions = self.getPrdictions()
            print("Predictions Calculated")

            # 4. Save the predictions data to a csv file
            self.saveResults()
            print("Save the results to a text file")
        else:
            print(FeatureEngineering.columnDetector(self.dataTrainX[0]))


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
            if flag == "train X":
                output = lambda data: [float(val) if (val != '') else None for val in data[0:79]]
            elif flag == "train Y":
                output = lambda data: [float(val) for val in data[79]]
            elif flag == "test X":
                output = lambda data: [float(val) for val in data[1].split(' ')]
            return output(row)
        except:
            print("Mapper function not working properly, Issue in Flag", flag)


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
                        xTrain.append(row)
                        yTrain.append(row)
                    return np.array(xTrain), np.array(yTrain)
                elif ("test" in fileName):
                    xTest = []
                    for row in reader:
                        xTest.append(row)
                    return np.array(xTest)
        except:
            print("File not present or conversion issue from list to array while Loading")


    def mlModel(self):
        '''
        Here we can use Linear Regression
        technique to find out the the sales
        price.
        '''
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        return regr.fit(self.dataTrainX, self.dataTrainY)


    def getPrdictions(self):
        '''
        Here we are just using the already
        built model and using it for predicting
        the house prices as results.
        '''
        # Make predictions using the testing set
        houses_y_pred = self.model.predict(self.dataTestX)
        return houses_y_pred


    def saveResults(self, fileName):
        '''
        Save the predicted house prices
        to a csv file.
        '''
        with open(fileName) as myFile:
            myFile.write(self.pred)
            myFile.close()


if __name__ == '__main__':
    obj = HousePricePrediction(True, "data/train.csv", "data/test.csv")
