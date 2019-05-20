
import csv
import collections
import math
import numpy as np
import array as arr
import pandas as pd
from scipy.io import arff
import timeit


#fishing =-1
train_df = pd.read_csv("phishing_training.csv").drop('id', axis=1)
test_df = pd.read_csv("phishing_test.csv").drop('id', axis=1)
train_df.head()
test_df.head()
feature_set = train_df.drop('Result', axis=1)
result_set = train_df['Result'].values
test_featureset =test_df.drop('Result', axis=1)
testresult_set=test_df['Result'].values
def prio_probResult():
    pPhishing = -2
    pLeggite = -2
    count=0
    phishCount=0
    legCount=0
    for index, row in feature_set.iterrows():
        phish = result_set[index]
        if(phish==1):
            legCount=legCount+1
        else:
            phishCount=phishCount+1
        count=count+1
    pPhishing=phishCount/count
    pLeggite=legCount/count

    return pLeggite,pPhishing

#Conditional prob of (parameter/given legit)
#(sneezing(x)|flu(y))= sneezing+flu/flu
#column idn is the attribute feature
def ConditionalProb(x,y,columnInd):

    count=0
    sumxy=0

    for index, row in feature_set.iterrows():
        resy = result_set[index]
        if(row[columnInd]==x and resy==y):
            sumxy=sumxy+1
            count=count+1
        elif(resy==y):
            count = count + 1
    if(sumxy==0):
        sumxy=sumxy+1
        count=count+3
    return sumxy/count

def BayesCalc(testRow):

#calc top of equ and make the result 1
    global ConditionalProbList
    global ConditionalProbListValues
    global leg
    global phi


    x=1
    b=1
    #for index, row in feature_set.iterrows():
    for i, col in test_featureset.iteritems():

        a =[col[testRow],1,i]

        if(a in ConditionalProbList):
            for j in range(len(ConditionalProbList)):
                if(ConditionalProbList[j]==a):
                    xi=ConditionalProbListValues[j]
                    break
        else:

            xi=ConditionalProb(a[0],a[1],a[2])
            ConditionalProbList.append(a)
            ConditionalProbListValues.append(xi)
        x=x*xi

        #bi=ConditionalProb(col[testRow],-1,i)
        a = [col[testRow],-1,i]

        if (a in ConditionalProbList):
            for j in range(len(ConditionalProbList)):
                if (ConditionalProbList[j] == a):
                    bi = ConditionalProbListValues[j]
                    break
        else:

            bi = ConditionalProb(a[0],a[1],a[2])
            ConditionalProbList.append(a)
            ConditionalProbListValues.append(bi)

        b=b*bi

    top=x*leg
    bottom=b*phi+top
    return top/bottom

def prio_prob(columnInd):
    pZero = 0
    pOne = 0
    pNegOne=0
    count=0

    for index, row in feature_set.iterrows():

        if(row[columnInd]==0):
            pZero=pZero+1
        elif(row[columnInd]==-1):
            pNegOne=pNegOne+1
        else:
            pOne=pOne+1
        count=count+1
    pZero = pZero/count
    pOne = pOne/count
    pNegOne = pNegOne/count

    return pNegOne,pZero,pOne

def runTests():
    correct=0
    countcorrectleg = 0
    countbadleg = 0
    countcorrectphi = 0
    countbadphi = 0
    for i in range(len(testresult_set)-1):
        x = "false"

        testresult = testresult_set[i]
        test1 = BayesCalc(i)
        if(test1<=0.5):
            t1=-1
            if (testresult == t1):
                countcorrectphi = countcorrectphi + 1
            else:countbadphi=countbadphi+1

        else:
            t1=1
            if (testresult == t1):
                countcorrectleg = countcorrectleg + 1
            else:countbadleg=countbadleg+1
        if(testresult==t1):
            correct=correct+1
            x="true"
        print(test1, "\t", t1,"\t",x)
    print("confusion matrix:")
    print("\t-1\t1")
    print("-1\t",countcorrectphi,"\t",countbadphi)
    print("1\t",countbadleg,"\t",countcorrectleg)
    print("")
    print("Accuracy= ",(countcorrectphi+countcorrectleg)/(countcorrectphi+countcorrectleg+countbadleg+countbadphi)*100)

leg,phi=prio_probResult()

ConditionalProbList=[]
ConditionalProbListValues=[]
start = timeit.timeit()

runTests()
end = timeit.timeit()
print ("Runttime:",end - start)
# print("confusion matrix:")
# print("\t-1\t1")
# print("-1\t1001\t361")
# print("1\t330\t765")
# print("")
# print("Accuracy= 71.88%")

