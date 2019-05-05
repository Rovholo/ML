from math import *
import numpy as np
from copy import deepcopy
from random import choice

class Tree():
    def __init__(self,v):
        self.attribute = v
        self.values = {}

    def traverse(self,dictionary):
        try:
            t = self.values[ dictionary[self.attribute] ]
        except:
            t = choice([ self.values[i] for i in list(self.values.keys()) if i not in dictionary ])
        if isinstance(t, Tree):
            return t.traverse(dictionary)
        else:
            return t

def readfile(file):
    f,attributes,table = open(file),{},{}
    for s in f:
        if len(s) > 2 and s.find("@relation") == -1 and s.find("@data") == -1:
            if s.find("@attribute") != -1:
                s1 = s.split()
                decider = s1[1]
                attributes.update( { decider : sorted(s1[3].split(",")) } )
                table.update( { decider : [] } )
            else:
                keys = list(attributes.keys())
                for i in range( len(keys) ):
                    table[keys[i]].append(s.split()[0].split(",")[i])
    return attributes,table,decider

def split_tables(table,attributes,pivot):
    tmp = [ [ i for i in range(len(table[pivot])) if table[pivot][i] == t ] for t in attributes[pivot] ] # collect indes of pivot values in table
    attributes_arr = [ dict([(d , []) for d in table]) for i in tmp ]
    tables = deepcopy(attributes_arr)
    decisions = []
    for i in range(len(tmp)):
        decisions.append( table[ pivot ][ tmp[i][0] ] )
        for d in table:
            for t in range(len(table[d])):
                if t in tmp[i]:
                    tables[i][d].append(table[d][t])
                    if table[d][t] not in attributes_arr[i][d]:
                        attributes_arr[i][d].append(table[d][t])
    return decisions,tables,attributes_arr

def ID3(table,attributes,*pivot):
    if pivot:
        del table[pivot[0]]
        del attributes[pivot[0]]

    if(len(attributes[decider]) == 1):
        return attributes[decider][0]
    elif len(table) == 1:
        return attributes[decider][ np.argmax( [ table[decider].count(i) for i in attributes[decider] ] ) ]

    H = lambda x: -sum([i*log(i,2) for i in x if i]) #calculates entropy
    G = lambda x: HD - ( 1/len(table[decider]) )*sum(np.array(D)*np.array(entropy)) #calculates gain in entropy

    HD = H( [ table[decider].count(i)/len(table[decider]) for i in attributes[decider] ] ) #calculate entopy of whole table
    maxgain = 0
    for d in table:
        if d != decider:
            entropy,D = [],[]
            for c in attributes[d]:
                tmp = []
                for y in attributes[decider]:
                    tmp.append( len( [ i for i in range(len(table[d])) if(table[d][i] == c and table[decider][i] == y) ] )/table[d].count(c) ) # calculate probabilities and add them to tmp
                entropy.append( H( tmp ) ) #calculate etropy and add it to array
                D.append( table[d].count(c) ) #calculate D values and add them to array
            gain = G(entropy)

            if gain > maxgain: #if gain is max set attribute as pivot
                maxgain = gain
                pivot = d
    if maxgain == 0:
        return table[decider][ np.argmax( [ table[decider].count(i) for i in attributes[decider] ] ) ]

    decisions,tables,attributes_arr = split_tables(table,attributes,pivot)
    node = Tree(pivot)
    for i in range(len(tables)):
        node.values.update( { decisions[i] : ID3(tables[i],attributes_arr[i],pivot) })
    return node

def top_branch(root):
    for t in root.values.values():
        if isinstance(t, Tree):
            return False
    return True

def remove_inconsistency(values, attributev):
    values = list(dict.fromkeys(values))
    if len(values) > len(attributev):
        values = [i for i in values if isinstance(i,Tree) or i in attributev ]
        if len(values) > len(attributev):
            values = [ i for i in values if i in attributev ]
    elif len(values) < len(attributev):
        [ attributev.remove(i) for i in attributev if len(values) < len(attributev) and i not in values ]
    return values,attributev

def validation(root,table,attributes):

    if not isinstance(root,Tree):
        return root

    values,attributes[root.attribute] = remove_inconsistency(list(root.values.values()), attributes[root.attribute])
    decisions,tables,attributes_arr = split_tables(table,attributes,root.attribute)

    nodes = [ validation(values[i],tables[i],attributes_arr[i]) for i in range(len(values)) ]
    err = error(root,table,attributes)
    for i in nodes:
        if not isinstance(i,Tree):
            err2 = error(i,table,attributes)
            if err > err2:
                root = i
                err = err2
    return root

def error(root,table,attributes):
    if isinstance(root,Tree):
        a = classify(root,table,attributes)
        return len( [ i for i in range(len(a)) if a[i] != table[decider][i] ] ) / len(table[decider])
    else:
        return len([ i for i in table[decider] if i != root ]) / len(table[decider])

def classify(root,table,attributes):
    return [ root.traverse(dict( [ (d, table[d][i]) for d in table if d != decider ] )) for i in range(len(table[decider])) ]

attributes,table,decider = readfile("Training Dataset.arff")
root = ID3(table,attributes)

attributes1,table1 = readfile("old.arff")[:2]
node = validation(root,table1,attributes1) #prune with validation data and set thr root of prunrd tree as node

print("error root rel old",error(root,table,attributes)) #calculate error in training data
print("error root rel new",error(root,table1,attributes1)) #calculate error in validation data

print("error node rel old",error(node,table,attributes)) #calculate error in training data after pruning
print("error node rel new",error(node,table1,attributes1)) #calculate error in validation data after pruning
