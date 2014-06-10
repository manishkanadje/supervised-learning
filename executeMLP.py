# This program reads a Neural Network from a .csv file and input data.
# It uses those weights for finding the predicted class for a given input
# sample. It also creates an image file within [0, 0] to [1, 1] grid in order
# to produce the classification region.
#
# Author : Manish Kanadje
# Date : April 28, 2014

import csv
import math
import matplotlib.pyplot as plt
import sys

# Reads the input data from .csv file
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

# This function returns sigmoid output for an input sample using logistic 
# regression. This node is used after the input layer and hidden layer. Output 
# of this node is an input to the next layer.
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

# Creates the required weight matrices using the given input .csv file
def createWeightMatrices(weightFile, inputNodes, hiddenNodes, outputNodes):
    allWeights = []
    inputWeights = []
    hiddenWeights = []
    with open(weightFile, 'rU') as weightFile:
        reader = csv.reader(weightFile, dialect = csv.excel)
        for rows in reader:
            allWeights.append(rows)
        for k in range(len(allWeights)):
            allWeights[k] = map(float, allWeights[k])
    for i in range(hiddenNodes + 1):
        inputWeights.append(allWeights[i])    
    for i in range(hiddenNodes + 1, len(allWeights)):
        hiddenWeights.append(allWeights[i])
    return inputWeights, hiddenWeights

def findMaxResult(finalOutput):
    max = 0
    maxIndex = 0
    for i in range(len(finalOutput)):
        if (finalOutput[i] > max):
            max =finalOutput[i]
            maxIndex = i
    return maxIndex

# Returns the sigmoid of an input value
def sigmoid(hx):
    return (1 / (1 + pow(math.e, -hx)));

# Creates an image of classification regions based on the input values within a 
# grid of (0,0) to (1,1). 
def createDecisionBoundary(inputWeights, hiddenWeights):
    xco_ordinate = 0
    while (xco_ordinate <= 1):
        yco_ordinate = 0
        while (yco_ordinate <= 1):
            input = [xco_ordinate ,yco_ordinate]
            hiddenLayerOutput = regressorResult(inputWeights, input)
            finalOutput = regressorResult(hiddenWeights, hiddenLayerOutput)
            maxIndex = findMaxResult(finalOutput)
            if (maxIndex == 0):
                plt.plot(xco_ordinate, yco_ordinate, "r.", ms = 10.0)
            elif (maxIndex == 1):
                plt.plot(xco_ordinate, yco_ordinate, "k.", ms = 10.0)
            elif (maxIndex == 2):
                plt.plot(xco_ordinate, yco_ordinate, "b.", ms = 10.0)
            else:
                plt.plot(xco_ordinate, yco_ordinate, "m.", ms = 10.0)
            yco_ordinate += 0.01
        xco_ordinate += 0.01
    plt.show()

# Prints the final profit and confusion matrix in the standard output
def printConfusionMatrix(confusionMatrix, priceMatrix):
    finalPrice = 0
    for i in range(len(confusionMatrix)):
        for j in range(len(confusionMatrix[0])):
            finalPrice += confusionMatrix[i][j] * priceMatrix[i][j]
    print "Total Profit :", finalPrice
    nameList = ['', 'Bolt', 'Nut', 'Ring', 'Scrap']
    print 'Confusion Matrix : '
    for i in range(len(nameList)):
        print '%7s' % nameList[i],
    print ''
    for i in range(len(confusionMatrix)):
        print '%7s' % nameList[i + 1],
        for j in range(len(confusionMatrix[0])):
            print '%7s' % confusionMatrix[i][j],
        print ''

def main():
    if (len(sys.argv) != 3):
        print "That's not the correct Input"
        print "Enter something like : python executeMLP.py test_data.csv weights1.csv"
    else:
        priceMatrix = [[0.20, -0.07, -0.07, -0.07], [-0.07, 0.15, -0.07, -0.07,], \
                        [-0.07, -0.07, 0.05, -0.07], [-0.03, -0.03, -0.03, -0.03]]
        inputFile = sys.argv[1]
        weightFile = sys.argv[2]
        expectedClasses = [1, 2, 3, 4]
        inputNodes = 2
        hiddenNodes = 5
        outputNodes = 4
        confusionMatrix = [[0 for col in range(len(expectedClasses))] \
                           for row in range(len(expectedClasses))]
        inputWeights, hiddenWeights = createWeightMatrices(weightFile, \
                                                           inputNodes, \
                                                           hiddenNodes, \
                                                           outputNodes)
        inputData = readInputData(inputFile)  
        correct = 0  
        for i in range(len(inputData)):
            initInput = inputData[i]
            actualOutput = initInput[len(initInput) - 1]
            inputRow = [initInput[0], initInput[1]]
            errorIndex = expectedClasses.index(actualOutput)
            hiddenLayerOutput = regressorResult(inputWeights, inputRow)
            finalOutput = regressorResult(hiddenWeights, hiddenLayerOutput)
            maxIndex = findMaxResult(finalOutput)
            #print "Example :", i
            #print "Actual Class : ", actualOutput
            #print "Predicted Class :", expectedClasses[maxIndex]
            if (actualOutput == expectedClasses[maxIndex]):
                correct += 1
            confusionMatrix[maxIndex][expectedClasses.index(actualOutput)] \
                += 1
        print "Correct Classification :", correct
        print "Incorrect Classification : ", (len(inputData) - correct)
        print "Recognition Rate : ", (correct + 0.0)/(len(inputData))
        createDecisionBoundary(inputWeights, hiddenWeights)
        printConfusionMatrix(confusionMatrix, priceMatrix)

main()
