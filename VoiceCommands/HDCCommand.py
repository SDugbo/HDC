from MFCC import MFCC
import numpy as np
from numpy import linalg as LA
import pickle
import copy
# import dataset



testData = MFCC.getTestFiles()
trainData = MFCC.getTrainFiles()

testLabels = trainLabels[5238:]
trainLabels = trainLabels[:5237]

trainData = np.matrix(trainData)
testData = np.matrix(testData)
num_trainData, feature_size = trainData.shape
# Generates a bipolar random matrix size D*feature_size
def genRandomMatrix(D, feature_size):
    random_matrix = np.random.rand(D, feature_size)
    return np.where(random_matrix>0.5, 1, -1)
def binarizeMatrix(m):
    return np.where(m>0, 1, -1)
# Encode data by dot multiplying random matrix and inputData
# Inputs: inputData(num_trainData*feature_size), random_matrix(D*feature_size)
# H(D*1) = random_matrix(D*feature_size) * data.T(feature_size*1)
# H.T is a 1*D numpy matrix
# return: encoded_data, a list of num_trainData numbers of 1*D numpy matrices
def encodeData(inputData, random_matrix):
    encoded_data = []
    for data in inputData:
        H = binarizeMatrix(np.dot(random_matrix, data.T))
        encoded_data.append(H.T[0])
    return encoded_data
# Trains the encoded data by simply adding the vector of same class
# Inputs: classHVs(dict), encoded_data(list of numpy matrices), trainLabels(list)
# return: classHVs(dict)
def train(classHVs, encoded_data, trainLabels):
    for i in range(len(trainLabels)):
        if trainLabels[i] in classHVs:
            classHVs[trainLabels[i]] += encoded_data[i]
        else:
            classHVs[trainLabels[i]] = encoded_data[i]
    return classHVs
def cosAngle(u, v):
    return np.dot(u,v)/(LA.norm(u)*LA.norm(v))
# Verfication and Retrain n times
# Return: binarized_classHVs
def retrain(nTims, encoded_data, classHVs, trainLabels):
    binarized_classHVs = {}
    for n in range(nTimes):
        for key in classHVs:
            binarized_classHVs[key] = binarizeMatrix(classHVs[key])
        correct = 0
        for i in range(len(encoded_data)):
            maxAngle = -1
            for label in classHVs:
                angle = cosAngle(encoded_data[i], binarized_classHVs[label])
                if angle > maxAngle:
                        maxAngle = angle
                        predicted_label = label
            if predicted_label == trainLabels[i]:
                correct += 1
            else:
                classHVs[trainLabels[i]] = np.add(classHVs[trainLabels[i]], encoded_data[i])
                classHVs[predicted_label] = np.subtract(classHVs[predicted_label], encoded_data[i])
        print("Verification accuracy: " + str(correct/len(encoded_data)) +
              ". Finished retraining " + str(n+1) + " times")
    return binarized_classHVs
def test(encoded_test, binarized_classHVs):
    correct = 0
    for i in range(len(testData)):
        maxAngle = -1
        for label in classHVs:
            angle = cosAngle(encoded_test[i], binarized_classHVs[label])
            if angle > maxAngle:
                    maxAngle = angle
                    predicted_label = label
        if predicted_label == testLabels[i]:
            correct += 1
    return correct/len(testData)
D = 5000
random_matrix = genRandomMatrix(D, feature_size)
encoded_data = encodeData(trainData, random_matrix)
classHVs = {}
classHVs = train(classHVs, encoded_data, trainLabels)
nTimes = 20
print("Finished training, starting to retrain " + str(nTimes) + " times")
binarized_classHVs = retrain(nTimes, encoded_data, classHVs, trainLabels)

encoded_test = encodeData(testData, random_matrix)
accuracy = test(encoded_test, binarized_classHVs)
print("Accuracy: " + str(accuracy))
Accuracy: 0.917