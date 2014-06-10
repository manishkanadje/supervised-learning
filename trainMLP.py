####################################################################################
# This program trains the Multilayer Perceptron using the backpropagation
# algorithm. Each weight is updated after one instance has been processed. It 
# creates the plot of Sum of Squared Error by calculating the error at the end of
# each epoch. The weights are saved after 10, 100, 1000, 10000 epochs. It produces
# 4 .cvs files containing the weights.
# 
# Author : Manish Kanadje
# Date :  April 25, 2014
####################################################################################

import csv
import math
import random
import pickle
import matplotlib.pyplot as plt
import math
import sys

##########################################
# Reads the input data from .csv file
##########################################

def readInputData(inputFile):
    inputData = []
    with open (inputFile, "rU") as trainFile:
        trainReader = csv.reader(trainFile)
        for rows in trainReader:
            if (rows != []):
                inputData.append(rows)
        for i in range(len(inputData)):
            inputData[i] = map(float, inputData[i])
    return inputData

#################################################################################
# This function returns sigmoid output for an input sample using logistic 
# regression. This node is used after the input layer and hidden layer. Output 
# of this node is an input to the next layer.
#################################################################################

def regressorResult(weightMatrix, input):
    bias = 1
    regressorOutput = []
    for i in range(len(weightMatrix)):
        cost  = weightMatrix[i][0] * bias;
        for k in range(len(input)):
            cost = cost + weightMatrix[i][k+1] * input[k]
        cost = sigmoid(cost)
        regressorOutput.append(cost)
    return regressorOutput

##################################
# Returns the sigmoid of a value
##################################
def sigmoid(hx):
    return (1 / (1 + pow(math.e, -hx)));

##############################################################################
# This function returns the input for final output layer. Weights depend on 
# the number of nodes present in the hidden layer.
##############################################################################

def regresorHidden(weights, input):
    bias = 1
    cost = weights[0] * bias
    for i in range(len(input)):
        cost = cost + input[i] * weights[i + 1]
    print "input for output layer"

##############################################################
# Calculates the error at each node of the final output layer 
##############################################################

def calculateOutputError(nodeValue, input, actualOutput, errorIndex):
    errorList = []
    for i in range(len(nodeValue)):
        expectedValue = 0
        if (i == errorIndex):
            expectedValue = 1
        error = (nodeValue[i] - expectedValue) * nodeValue[i] * \
            (1 - nodeValue[i])
        errorList.append(error)
    return errorList

#########################################################
# Calculates the error at each node of the hidden layer 
#########################################################

def calculateHiddenError(nodeValues, previousError, weightMatrix):
    errorList = []
    for i in range(len(nodeValues)):
        error = 0
        for j in range(len(previousError)):
            error = error + weightMatrix[j][i+1] * previousError[j]
        error  = error * nodeValues[i] * (1  - nodeValues[i])
        errorList.append(error)
    return errorList

# Updates the weights of nodes in any layer (hidden or output) based on the given
# error present at that layer    
def weightUpdate(weightMatrix, errorList, nodeValues, alpha):
    updatedWeightMatrix = [[0 for col in range(len(weightMatrix[0]))] \
                            for row in range(len(weightMatrix))]
    for i in range(len(weightMatrix[0])):
        for j in range(len(errorList)):
            updatedWeightMatrix[j][i] = weightMatrix[j][i] - alpha * \
                errorList[j] *  nodeValues[i]
    return updatedWeightMatrix
    print "update the weights based on the calculated error"

# Initializes weights between [-1, 1] randomly    
def initializeWeights(inputNodes, nextLayerNodes):
    weightMatrix = [[0 for col in range(inputNodes + 1)] for row in \
                     range(nextLayerNodes)]
    for i in range(len(weightMatrix)):
        for j in range(len(weightMatrix[0])):
            weightMatrix[i][j] = random.uniform(-1, 1)
    return weightMatrix

# Creates a weight file in the .csv format which stores the weights for all the nodes
# present in input or hidden layer. This file is created from 2 pickle files
def createWeightFiles():
    retrieveList = [0, 10, 100, 1000, 10000]
    for i in range(len(retrieveList)):
        inputString = 'input' + str(retrieveList[i]) + '.p'
        hiddenString = 'hidden' + str(retrieveList[i]) + '.p'
        fileName = 'weights' + str(retrieveList[i]) + '.csv'
        inputList = pickle.load(open(inputString, 'rb'))
        hiddenList = pickle.load(open(hiddenString, 'rb'))
        with open(fileName, 'wb') as weightFile:
            writer = csv.writer(weightFile)
            for i in range(len(inputList)):
                writer.writerow(inputList[i])
            for i in range(len(hiddenList)):
                writer.writerow(hiddenList[i])

def findMaxResult(finalOutput):
    max = 0
    maxIndex = 0
    for i in range(len(finalOutput)):
        if (finalOutput[i] > max):
            max =finalOutput[i]
            maxIndex = i
    return maxIndex

# Plots the curve of Sum of Squared Error vs Number of Epochs
def plotSquaredError(squaredErrorList):
    x_axis = [0] * len(squaredErrorList)
    for i in range(len(squaredErrorList)):
        x_axis[i] = i
    plt.plot(x_axis, squaredErrorList)
    plt.xlabel("Number of epochs")
    plt.ylabel("Total Squared Error")
    plt.show()

# Calculates the SSE for given input dataset using given MLP weights    
def calculateSquaredError(inputData, inputWeights, hiddenWeights,\
                             expectedClasses):
    error = 0
    for i in range(len(inputData)):
        initInput = inputData[i]
        actualOutput = initInput[len(initInput) - 1]
        inputRow = [initInput[0], initInput[1]]
        errorIndex = expectedClasses.index(actualOutput)
        hiddenLayerOutput = regressorResult(inputWeights, inputRow)
        finalOutput = regressorResult(hiddenWeights, hiddenLayerOutput)
        outputErrorList = calculateOutputError(finalOutput, inputRow, \
                                                 actualOutput, errorIndex) 
        for i in range(len(outputErrorList)):
            error += math.pow(outputErrorList[i], 2)
    return error
        
def main():
    if (len(sys.argv) != 2):
        print "That's not the correct Input"
        print "Enter something like : python trainMLP.py train_data.csv"
    else:
        inputFile = sys.argv[1]
        alpha = 0.1
        expectedClasses = [1.0, 2.0, 3.0, 4.0]
        hiddenNodes = 5
        inputData = readInputData(inputFile)
        pickleList = [10, 100, 1000, 10000]
        squaredErrorList = []
    
        inputWeights = initializeWeights(2, hiddenNodes + 1)
        hiddenWeights = initializeWeights(hiddenNodes + 1, 4)
        pickle.dump(inputWeights, open('input0.p', 'wb'))
        pickle.dump(hiddenWeights, open('hidden0.p', 'wb'))
        for iterationCount in range(10000):
            for i in range(len(inputData)):
                initInput = inputData[i]
                actualOutput = initInput[len(initInput) - 1]
                inputRow = [initInput[0], initInput[1]]
                errorIndex = expectedClasses.index(actualOutput)
                hiddenLayerOutput = regressorResult(inputWeights, inputRow)
                finalOutput = regressorResult(hiddenWeights, hiddenLayerOutput)
                outputErrorList = calculateOutputError(finalOutput, inputRow, \
                                                       actualOutput, errorIndex)
                hiddenErrorList = calculateHiddenError(hiddenLayerOutput, \
                                                       outputErrorList, \
                                                       hiddenWeights)
                hiddenLayerOutput.insert(0, 1)
                hiddenWeights = weightUpdate(hiddenWeights, outputErrorList, \
                                             hiddenLayerOutput, alpha)
                inputRow.insert(0, 1)
                inputWeights = weightUpdate(inputWeights, hiddenErrorList, \
                                            inputRow, alpha)
                maxIndex = findMaxResult(finalOutput)
            error = calculateSquaredError(inputData, inputWeights, hiddenWeights,\
                                        expectedClasses)
            squaredErrorList.append(error)
            if ((iterationCount + 1) in pickleList):
                inputString = 'input' +  str(iterationCount + 1) + '.p'
                hiddenString = 'hidden' +  str(iterationCount + 1) + '.p'
                pickle.dump(inputWeights, open(inputString, 'wb'))
                pickle.dump(hiddenWeights, open(hiddenString, 'wb'))
        createWeightFiles()
        plotSquaredError(squaredErrorList)
            
main()
