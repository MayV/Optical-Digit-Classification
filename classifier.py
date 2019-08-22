import pandas as pd
import numpy as np


#PCA Algorithm.
def PCA(A, threshold):
    mean = A.sum(axis=0)/len(A)
    X = A-np.tile(mean, (len(A), 1))
    covX = (X.T*X)/(len(A)-1)
    eigenValues, eigenVectors = np.linalg.eig(covX) #w[i] contains eigen value and v[:,i] contains eigen vector corresponding to w[i]
    print("Eigen values and respective Eigen vectors:")
    #print(eigenValues)
    #print(eigenVectors)
    sortedIdx = eigenValues.argsort()[::-1]
    threshold = threshold
    numPC = 0 #number of principal components.
    accuracy = 0 #amount of data retained.
    PC = np.zeros((eigenVectors.shape[0],0))
    for i in range(len(sortedIdx)):
        numPC += 1
        accuracy += eigenValues[sortedIdx[i]]/sum(eigenValues)
        PC = np.append(PC, eigenVectors[:,sortedIdx[i]], axis=1)
        if accuracy>=threshold:
            break;
    return PC


#KNN Algorithm.
def KNN(inX,trainMat,labels,k):
    import operator
    trainMatSize = trainMat.shape[0]
    diffMat = np.tile(inX, (trainMatSize,1)) - trainMat
    sqDiffMat = np.square(diffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    distances = np.sqrt(sqDistances)
    sortedDistIndices = distances.T.argsort()
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[0,i],0]
        #print(voteLabel)
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


#Load Training Dataset.
trainingData = pd.read_csv('optdigits.tra', header=None)
trainingData = np.matrix(trainingData)
trainingDataSet = trainingData[:,0:64]
trainingLabels = trainingData[:,64]


#Apply PCA and transform data to lower space.
PC = PCA(trainingDataSet, 0.9)
Y = trainingDataSet*PC


#Load Testing Dataset.
testingData = pd.read_csv('optdigits.tes', header=None)
testingData = np.matrix(testingData)
testingDataSet = testingData[:,0:64]
testingLabels = testingData[:,64]


#Classify
results = []
errorCount = 0
for i in range(len(testingDataSet)):
    vec = testingDataSet[i]*PC
    pred = KNN(vec, Y, trainingLabels, 100)
    print(testingLabels[i,0], pred)
    if pred != testingLabels[i,0]:
        errorCount += 1
    results.append(pred)
accuracy = (len(testingDataSet)-errorCount)/len(testingDataSet)
print(accuracy)