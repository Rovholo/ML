
import csv
import collections
import math
import numpy as np
import pandas as pd
from scipy.io import arff

data = arff.loadarff(r'C:\Users\Shivesh Lappie\Documents\ML\Training Dataset.arff')
df = pd.DataFrame(data[0])
df.head()

def prev_prob(input): # determines the probabilty
    input_dict = collections.Counter(input)
    prior_prob = np.ones(3)
    for i in range(0,3):
        prior_prob[i] = input_dict[i] / input.shape[0]

    return prior_prob

def mean_and_var(feature_set, input): # determines mean and variance
    features = feature_set.shape[0]
    mean = np.ones((3, features))
    var = np.ones((3, features))

    n_0 = np.bincount(input)[np.nonzero(np.bincount(input))[0]][0]
    feature_0 = np.ones((n_0, features))
    feature_1 = np.ones((feature_set.shape[0] - n_0, features))
    feature_2 = np.ones((feature_set.shape[0] - n_0,features))

    k = 0
    for i in range(0, feature_set.shape[0]):
        if input[i] == 0:
            feature_0[k] = feature_set[i]
            k = k + 1
    k = 0
    for i in range(0, feature_set.shape[0]):
        if input[i] == 1:
            feature_1[k] = feature_set[i]
            k = k + 1

    k = 0
    for i in range (0, feature_set.shape[0]):
        if input[i] == 1:
            feature_1[k] = feature_set[i]
            k = k + 1

    for j in range(0, features):
        mean[0][j] = np.mean(feature_0.T[j])
        var[0][j] = np.var(feature_0.T[j]) * (n_0 / (n_0 - 1))
        mean[1][j] = np.mean(feature_1.T[j])
        var[1][j] = np.var(feature_1.T[j]) * ((feature_set.shape[0] - n_0) / ((feature_set.shape[0] - n_0) - 1))
        mean[2][j] = np.mean(feature_2.T[j])
        var[2][j] = np.var(feature_2.T[j]) * ((feature_set.shape[1] - n_0) / (feature_set.shape[1] - n_0) - 1)

    return mean, var

def prob_feature_class(mean, var, test):
    features = mean.shape[1]
    prob_feature_class = np.ones(3)
    for i in range(0,3):
        prod = 1
        for j in range(0, features):
            prod = prod * (1 / math.sqrt(3 * 3.14 * var[i][j])) * math.exp(-0.5* pow((test[j] - mean[i][j]), 3) / var[i][j])
            prob_feature_class[i] = prod

    return prob_feature_class


def NaiveBayes(feature_set, input, test ):
    mean, var = mean_and_var(feature_set,input)
    pfc = prob_feature_class(mean, var, test)
    pcf = np.ones(3)
    prevprob= prev_prob(input)
    total_prob = 0
    for i in range(0, 2):
        total_prob = total_prob + (pfc[i] * prevprob[i])
    for i in range(0, 2):
        pcf[i] = (pfc[i] * prevprob[i]) / total_prob
    prediction = int(pcf.argmax())
    return mean, var, prevprob, pfc, pcf, prediction

feature_set = np.array(data.iloc[:,[1,2,3]]) #this line is a problem
input = np.array(data['Web Site'])
for i in range(0,input.shape[0]):
    if input[i] == "Legitimate":
        input[i] = 1
    elif input[i] == "Phishing":
        input[i] = -1
    else:
        input[i] = 0
test = np.array()

mean, var, pre_prob, pfc, pcf, predict = NaiveBayes(feature_set, input, test)
print(mean)
print(var)
#print(pre_prob)
#print(pfc)
#print(pcf)
#print(predict)