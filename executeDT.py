# This program executes the random forest decision tree. The forest is recreated from
# given input file. It predicts the class for given test dataset. It also produces
# an image of (0,0) to (1,1) input grid, displaying the different regions according
# to the classes.
#
# Author : Manish M Kanadje
# Date : April 29, 2014

import pickle
import csv
import matplotlib.pyplot as plt
import copy
import sys

# Creates the inputData list from given .csv file
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

# Creates the random forest list using the given pickle .p file
def getRandomForest(pickleFile):
    randomForest = pickle.load(open(pickleFile, 'rb'))
    return randomForest
        
# Return the output for a given instance using all the trees present in the random
# forest. It returns the final class values based on maximum probability
def getRandomForestOutput(inputRow, randomForest):
    probabilityList = []
    for j in range(len(randomForest)):
        tree = randomForest[j]
        index = 0
        iterationValue = 0
        while (iterationValue != None):
            attribute = tree[index][0]
            if (inputRow[attribute] > tree[index][1]):
                index = 2 * index + 2
                if (tree[index] != None):
                    iterationValue = tree[index][0]
                else:
                     index -= 1
                     iterationValue = tree[index][0]
            else:
                index = 2 * index + 1
                if (tree[index] != None):
                    iterationValue = tree[index][0]
                else:
                     index += 1
                     iterationValue = tree[index][0]
        probabilityVector = tree[index][1]
        probabilityList.append(copy.deepcopy(probabilityVector))
    finalVector = [0] * len(probabilityList[0])
    for k in range(len(probabilityList)):
        for l in range(len(probabilityList[0])):
            finalVector[l] += probabilityList[k][l]/len(randomForest)
    classValue = finalVector.index(max(finalVector)) + 1.0
    return classValue

# Creates the classification region for a given random forest using input points
# between [0,0] and [1,1] grid
def printDecisionMap(randomForest):
    x_value = 0.0
    while (x_value <= 1.0):
        y_value = 0.0
        while (y_value <= 1.0):
            inputRow = [x_value, y_value]
            predictedClass = getRandomForestOutput(inputRow, randomForest)
            if (predictedClass == 1):
                plt.plot(x_value, y_value, "r.", ms = 10.0)
            elif (predictedClass == 2):
                plt.plot(x_value, y_value, "k.", ms = 10.0)
            elif (predictedClass == 3):
                plt.plot(x_value, y_value, "b.", ms = 10.0)
            else:
                plt.plot(x_value, y_value, "m.", ms = 10.0)
            y_value += 0.01
        x_value += 0.01
    plt.show()

# Prints the confusion matrix and final profit obtained in the standard output            
def printConfusionMatrix(confusionMatrix, priceMatrix):
    finalPrice = 0
    for i in range(len(confusionMatrix)):
        for j in range(len(confusionMatrix[0])):
            finalPrice += confusionMatrix[i][j] * priceMatrix[i][j]
    print "Total Profit :", finalPrice 
    nameList = ['', 'Bolt', 'Nut', 'Ring', 'Scrap']
    print 'Confusion Matrix :'
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
        print "Enter something like : python executeDT.py test_data.csv tree0.p"
    else:
        priceMatrix = [[0.20, -0.07, -0.07, -0.07], [-0.07, 0.15, -0.07, -0.07,], \
                        [-0.07, -0.07, 0.05, -0.07], [-0.03, -0.03, -0.03, -0.03]]
        inputFile = sys.argv[1]
        pickleFile = sys.argv[2]
        expectedClasses = [1.0, 2.0, 3.0, 4.0]
        confusionMatrix = [[0 for row in range(len(expectedClasses))] \
                           for col in range(len(expectedClasses))]
        randomForest = getRandomForest(pickleFile)
        inputData = readInputData(inputFile)
        correct = 0
        for i in range(len(inputData)):
            inputRow = inputData[i]
            classValue = getRandomForestOutput(inputRow, randomForest)
            confusionMatrix [(int)(classValue - 1)][(int) (inputRow[2] - 1)] += 1
            #print "Instance :", i
            #print "Actual Class", inputRow[2]
            #print "Predicted Class", classValue
            if (classValue == inputRow[2]):
                correct += 1
        print "Correct Classification : ", correct
        print "Incorrect Classification : ", (len(inputData) - correct)
        print "Recognition Rate : ", (correct + 0.0)/(len(inputData))
        printDecisionMap(randomForest)
        printConfusionMatrix(confusionMatrix, priceMatrix)
    
main()
