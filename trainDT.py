# This program trains the random forest decision tree algorithm based on the 
# given input dataset. It produces the sum of the squared error after each
# decision tree ensemble is added to the random forest. The decision tree is a binary
# tree. It has been implemented using List structure. i.e. if n is the parent then
# 2*n + 1 is a left child and 2*n + 2 is a right child. 
#
# Author : Manish Kanadje
# Date : April 29, 2014

import csv
import random
import math
import copy
import matplotlib.pyplot as plt
import pickle
import sys

# Reads the input data from a given .csv file
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
# Calculates the entropy for a given data in numerator and denominator format 
def calculateEntropy(numerator, denominator):
    numerator += 0.0
    denominator += 0.0
    return ((-numerator/denominator) * math.log((numerator/denominator), 2))

# Generates k random split points within the rang of (0,1) as that is the range
# of input variable values    
def getKRandomSplits(k):
    kRandomSplits = []
    for i in range(k):
        kRandomSplits.append(random.uniform(0, 1))
    return kRandomSplits

# Calculates the entropies of given set of k random splits. It calls
# calculateEntropy function. log of 0 is not defined and we have consider it as 0.
# This function creates an output of 0 for log 0.        
def getEntropies(kRandomSplits, attributeNumber, remainingData):
    kEntropies = []
    totalInstances = len(remainingData)  
    if (totalInstances > 0):
        for splitValue in kRandomSplits:
            positiveDivision = [0, 0, 0, 0]
            negativeDivision = [0, 0, 0, 0]
            positiveInstance = 0.0
            negativeInstance = 0.0
            for i in range(len(remainingData)):
                if (remainingData[i][attributeNumber] > splitValue):
                    positiveInstance += 1
                    index = (int) \
                        (remainingData[i][len(remainingData[0]) - 1] - 1)
                    positiveDivision[index] += 1
                else:
                    negativeInstance += 1
                    index = (int) \
                        (remainingData[i][len(remainingData[0]) - 1] - 1)
                    negativeDivision[index] += 1
            if (positiveInstance == 0):
                positiveInstance += 0.00001
            if (negativeInstance == 0):
                negativeInstance += 0.00001
            for i in range(len(positiveDivision)):
                if (positiveDivision[i] == 0):
                    positiveDivision[i] = positiveInstance
                if (negativeDivision[i] == 0):
                    negativeDivision[i] = negativeInstance
            #print "Debug: math domain error"
            remainderPos = calculateEntropy(positiveDivision[0], positiveInstance) +\
                calculateEntropy(positiveDivision[1], positiveInstance) + \
                calculateEntropy(positiveDivision[2], positiveInstance) + \
                calculateEntropy(positiveDivision[3], positiveInstance)
            remainderNeg = calculateEntropy(negativeDivision[0], negativeInstance) + \
                calculateEntropy(negativeDivision[1], negativeInstance) + \
                calculateEntropy(negativeDivision[2], negativeInstance) + \
                calculateEntropy(negativeDivision[3], negativeInstance)
            remainder = positiveInstance / totalInstances * remainderPos + \
                    negativeInstance / totalInstances * remainderNeg
            kEntropies.append(remainder)
    return kEntropies

# Checks whether all the instances present in the input dataset have one single
# classification
def allSameClass(inputData):
    tempList = []
    for i in inputData:
        tempList.append(i[len(i) - 1])
    temp = tempList[0]
    for i in range(1, len(tempList)):
        if (temp != tempList[i]):
            return False, None
    return True, temp

# This function decides the value of the node in the decision tree. This value is
# stored as a list containing [attributeNumber, classValue]. attributeNumber is 
# None if no further classification occurs. classValue is the splitPoint for that
# attributeNumber at that node. If no further classification occurs, the classValue
# is a probability vector containing probabilities for all 4 classes.
# side is a boolean variable. Side is true if it is left node and false if it is a
# right node
def classNodeDecision(treeList, classValue, parent, attributeNumber, side, \
                         notTerminal):
    if (side):
        treeList[parent * 2 + 1] =  [attributeNumber, classValue]
        if (notTerminal):
            parent = parent * 2 + 1
            
    else:
        treeList[parent * 2 + 2] =  [attributeNumber, classValue]
        if (notTerminal):
            parent = parent * 2 + 2
    return parent


# This function creates the actual decision tree in a List and returns that list.
# It has breaking conditions for maximum height of tree, if the dataset is empty.
# It creats the decision tree based on the best entropy from k random splits for
# each variable.
def createEnsemble(inputData, numberOfAttributes, numberOfRandomSplits, \
                    treeList, height, parent, side, maxHeight):
    if (height < maxHeight):
        if (len(inputData) > 0):
            truth, value = allSameClass(inputData)
            if (truth and parent != None):
                valueList = [0, 0, 0, 0]
                valueList[(int) (value - 1)] = 1
                parent = classNodeDecision(treeList, valueList, parent, None, \
                                            side, False)
            else:
                rightData = []
                leftData = []
                splitList = []
                attributeNumberOne = 0
                kRandomSplitsOne = getKRandomSplits(numberOfRandomSplits)
                kEntropiesOne = getEntropies(kRandomSplitsOne, attributeNumberOne, \
                                                inputData)
                minEntropyOne = min(kEntropiesOne)
                bestSplitOne = kRandomSplitsOne[kEntropiesOne.index(minEntropyOne)]
                attributeNumberTwo = 1
                kRandomSplitsTwo = getKRandomSplits(numberOfRandomSplits)
                kEntropiesTwo = getEntropies(kRandomSplitsTwo, attributeNumberTwo, \
                                                 inputData)
                minEntropyTwo = min(kEntropiesTwo)
                bestSplitTwo= kRandomSplitsTwo[kEntropiesTwo.index(minEntropyTwo)]
                if (minEntropyOne < minEntropyTwo):
                    attributeNumber = attributeNumberOne
                    bestSplit = bestSplitOne
                else:
                    attributeNumber = attributeNumberTwo
                    bestSplit = bestSplitTwo
                for index in range(len(inputData)):
                    if (inputData[index][attributeNumber] > bestSplit):
                        rightData.append(copy.deepcopy(inputData[index]))        
                    else:
                        leftData.append(copy.deepcopy(inputData[index]))
                if (parent == None):
                    treeList[0] = [attributeNumber, bestSplit]
                    parent = 0
                else:
                    parent = classNodeDecision(treeList, bestSplit, parent, \
                                                attributeNumber, side, True)
                createEnsemble (rightData, numberOfAttributes, \
                            numberOfRandomSplits, treeList, \
                            height + 1, parent, False, maxHeight)
                createEnsemble (leftData, numberOfAttributes, \
                            numberOfRandomSplits, treeList, \
                            height + 1, parent, True, maxHeight)
    elif (height == maxHeight and len(inputData) > 0):
        probVector = getProbabilityVector(inputData)
        parent = classNodeDecision(treeList, probVector, parent, None, side, False)
    return treeList

# Returns the probability vector which is present at classification nodes. It 
# contains the probability for all classes.
def getProbabilityVector(inputData):
    tempList = []
    probabilityVector = [0, 0, 0, 0]
    countList = [0, 0, 0, 0]
    for i in inputData:
        tempList.append(i[len(i) - 1])
    for i in range(len(tempList)):
        countList[(int) (tempList[i] - 1)] += 1.0
    for i in range(len(countList)):
        probabilityVector[i] = countList[i]/len(inputData)
    return probabilityVector

# Calculates the Sum of Squared Error for all the instances using all the trees
# present in the given rondom forest
def calculateError(randomForest, inputData, expectedClasses):
    error = 0
    for i in range(len(inputData)):
        inputRow = inputData[i]
        votingList = []
        probabilityList = []
        for j in range(len(randomForest)):
            tree = randomForest[j]
            index = 0
            while (tree[index][0] != None):
                attribute = tree[index][0]
                if (inputRow[attribute] > tree[index][1]):
                    index = 2 * index + 2
                else:
                    index = 2 * index + 1
            probabilityVector = tree[index][1]
            probabilityList.append(copy.deepcopy(probabilityVector))
        finalVector = [0] * len(probabilityList[0])
        for k in range(len(probabilityList)):
            for l in range(len(probabilityList[0])):
                finalVector[l] += probabilityList[k][l]/len(randomForest)
        actualValue = [0] * len(probabilityList[0])
        actualValue[(int) (inputRow[2] - 1)] = 1
        totalError = 0
        for k in range(len(finalVector)):
            finalVector[k] = finalVector[k] - actualValue[k]
            totalError += math.pow(finalVector[k], 2)
        #totalError = math.pow(totalError, 2)
        error += totalError
    return error
        
# Creates the plot of SSE vs number of trees in the random forest
def plotSquaredError(errorList):
    x_axis = [0] * len(errorList)
    for i in range(len(errorList)):
        x_axis[i] = i
    plt.plot(x_axis, errorList)
    plt.xlabel("Number of Trees in the Forest")
    plt.ylabel("Total Squared Error")
    plt.show()


def main():
    if (len(sys.argv) != 2):
        print "That's not the correct Input"
        print "Enter something like : python trainDT.py train_data.csv"
    else:
        inputFile = sys.argv[1]
        expectedClasses = [1.0, 2.0, 3.0, 4.0]
        randomForest = []
        errorList = []
        pickleList = [1, 10, 100, 200, 400]
        maxHeight = 2
        numberOfAttributes = 2
        numberOfRandomSplits = 5
        inputData = readInputData(inputFile)
        for iterationCount in range(200):
            treeList = [None] * 15
            height = 0
            parent = None
            randomForest.append(createEnsemble(inputData, numberOfAttributes, \
                                               numberOfRandomSplits, \
                                               treeList, height, parent, True, \
                                                 maxHeight))
            error = calculateError(randomForest, inputData, expectedClasses)
            errorList.append(error)
            if ((iterationCount + 1) in pickleList):
                inputString = 'tree' +  str(iterationCount + 1) + '.p'
                pickle.dump(randomForest, open(inputString, 'wb'))
        plotSquaredError(errorList)
    
main()
