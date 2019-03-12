from array import array
from random import random

from nn.activations import sigmoid
from numpy import shape, ones


def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    dataIndex=range(m)
    weights=ones(n)
    for j in range(numIter):
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(intX,weights):
    prob=sigmoid(sum(intX*weights))
    if prob>0.5:return 1.0
    else:return 0.0

def colicTest():
    frTrain=open('hosrseColic.txt')
    frTest=open('horseColicText.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print('the error rate of this test is: %f'%errorRate)
    return errorRate

def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is: %f"%(numTests,errorSum/float(numTests)))
