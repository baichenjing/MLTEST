from numpy.matlib import zeros, shape, sign, multiply, mat, nonzero, exp


def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True;alphaPairsChanged=0
    while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            iter+=1
        else:
            nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
        if entireSet:entireSet=False
        elif (alphaPairsChanged==0):entireSet=True
    return oS.b,os.alphas
def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if kTup[0]=='lin':K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else: raise NameError('that Kernel is not recognized')
    return K

def loadImages(dirName):
    from os import listdir
    hwLabels=[]
    trainingFileList=listdir(dirName)
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr==9:hwLabels.append(-1)
        else:hwLabels.append(1)
        trainingMat[i,:]=img2vector('%s/%s'%(dirName,fileNameStr))
    return trainingMat,hwLabels
def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImages('trainingDigits')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    datMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV=labelMat[svInd]
    print("there are %d Support Vectors"% shape(sVs)[0])
    m,n=shape(datMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount+=1
    print("the tranning error rate is :%f"%(float(errorCount)/m))
    dataArr,labelArr=loadImages('testDigits')
    errorCount=0
    datMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    m,n=shape(datMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount+=1
    print("the test error rate is:%f"%(float(errorCount)/m))
