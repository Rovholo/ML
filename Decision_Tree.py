from math import *
import numpy as np
from copy import deepcopy
from random import choice

#Defines the tree class
class Tree():
    def __init__(self,v):
        self.attribute = v #atribute of each node
        self.values = {} #values the node can take
    def traverse(self,dictionary):
        try:
            t = self.values[ dictionary[self.attribute] ]
        except: #handle incosistency in data
            if dictionary[self.attribute] == '0' and '-1' in self.values:
                t = self.values['-1']
            elif dictionary[self.attribute] == '-1' and '0' in self.values:
                t = self.values['0']
            else:
                t = choice([ self.values[i] for i in list(self.values.keys()) if i not in dictionary ])
        if isinstance(t, Tree):
            return t.traverse(dictionary)
        else:
            return t

#reads in a file and create relevant dictionaries out of it
def readfile(file):
    f,attributes,table = open(file),{},{} #open our file and declare 2 dictionaries
    for s in f: #for each line in our file
        if len(s) > 2 and s.find("@relation") == -1 and s.find("@data") == -1: #if the data is not an empty space or the words @relations and @data
            if s.find("@attribute") != -1: #if the data starts with @relation
                s1 = s.split() #split the string with empty spaces
                decider = s1[1] #declare current class as results class
                attributes[decider] = s1[3].split(",") #store the @relations data as a dictinary
                table[decider] = [] #declare the table dictionary values as empty arrays
            else:
                keys = list(attributes.keys()) #make a list of the dictionary keys of attribues
                for i in range( len(keys) ): #for each key values
                    table[keys[i]].append(s.split()[0].split(",")[i]) #store the content of the class
    return attributes,table,decider #return the attributes dictionary,table dictionary and string for results class

#splits our table int different sub tables
def split_tables(table,attributes,pivot):
    tmp = [ [ i for i in range(len(table[pivot])) if table[pivot][i] == t ] for t in attributes[pivot] ] # collect indes of pivot values in table
    tables = [ dict([(d , []) for d in table]) for i in tmp ] #create array of empty table dictionaries
    attributes_arr = deepcopy(tables) # make copy of the array for storing classes and values they can take
    decisions = [] #declare an array for storing decisions to be made to access current branches(tables)
    for i in range(len(tmp)): #for each sub table/branch
        decisions.append( table[ pivot ][ tmp[i][0] ] ) #store the decion to be made to accesss it
        for d in table: #for each value in innitial table
            for t in range(len(table[d])): #for each value found in of the table at current class
                if t in tmp[i]: #if currentt value is found on indeces of current class
                    tables[i][d].append(table[d][t])
                    if table[d][t] not in attributes_arr[i][d]: #if the class value is not already stored
                        attributes_arr[i][d].append(table[d][t]) #store the value
    return decisions,tables,attributes_arr #return the decisions for acccesing branch, the sub tables and values the classes can take for each sub table

#performs the ID3 algorithm on current table
def ID3(table,attributes,*pivot):
    if pivot: #if the function was passed with variable 'pivot' element
        del table[pivot[0]] #delete the pivot column from table
        del attributes[pivot[0]] #delete the pivot attributes

    if(len(attributes[decider]) == 1): #if all decisions in our tsb;e ;lead to the same answer
        return attributes[decider][0] #return result/the answer
    elif len(table) == 1: #if we are out of classes to choose from
        return attributes[decider][ np.argmax( [ table[decider].count(i) for i in attributes[decider] ] ) ] #return the most occuring value on result column

    H = lambda x: -sum([i*log(i,2) for i in x if i]) #function for calculating entropy
    G = lambda x: HD - ( 1/len(table[decider]) )*sum(np.array(D)*np.array(entropy)) #function for calculating gain in entropy
    HD = H( [ table[decider].count(i)/len(table[decider]) for i in attributes[decider] ] ) #calculate entopy of whole table
    maxgain = 0 #variable for storing maximum gain in entropy
    for d in table:#for each attribute/class in table
        if d != decider: #if the class is not th eresults class
            entropy,D = [],[] #declare variables for storing entopy and D
            for c in attributes[d]: #for each value current class can take
                tmp = [] #create temporary arrar
                for y in attributes[decider]: #for ech value in the results class can take
                    tmp.append( len( [ i for i in range(len(table[d])) if(table[d][i] == c and table[decider][i] == y) ] )/table[d].count(c) ) # calculate probabilities and add them to tmp
                entropy.append( H( tmp ) ) #calculate etropy and add it to array
                D.append( table[d].count(c) ) #calculate D values and add them to array
            gain = G(entropy) #calculate gain in entropy

            if gain > maxgain: #if gain is max set attribute as pivot
                maxgain = gain
                pivot = d
    if maxgain == 0: #if there was no maximum gain
        return table[decider][ np.argmax( [ table[decider].count(i) for i in attributes[decider] ] ) ]

    decisions,tables,attributes_arr = split_tables(table,attributes,pivot) #split table int differnt peaces
    node = Tree(pivot) #declare a node for our tree
    for i in range(len(tables)): #for each class in table
        node.values.update( { decisions[i] : ID3(tables[i],attributes_arr[i],pivot) }) #add the node and its children
    return node #return current node

#runs validation data and prunes the tree
def validate(root,table,attributes):
    if not isinstance(root,Tree): #if the root is not of type tree
        return root #return the value of root

    nodes = [ validate(i,table,attributes) for i in root.values.values() ] #loop through branches of node, and call validate recursively
    errs = [error(i,table,attributes) for i in nodes] #calculate error in all sub branches
    mn = min(errs) #calculate the minimum of the errors of sub branches

    if mn < error(root,table,attributes): #if the error in branch is greater than that of a sub branch
        root =  nodes[errs.index(mn)] #replace the branch by the sub branch
    return root #return the branch

#calculates the error on current branch
def error(root,table,attributes):
    if isinstance(root,Tree): #if root if of type tree (is not a leaf)
        a = classify(root,table,attributes) #classify the data with current node
        return len( [ i for i in range(len(a)) if a[i] != table[decider][i] ] ) / len(table[decider]) #calulate the error
    else:
        return len([ i for i in table[decider] if i != root ]) / len(table[decider]) #calulate the error

#classifies the data using current branch
def classify(root,table,attributes):
    return [ root.traverse(dict( [ (d, table[d][i]) for d in table if d != decider ] )) for i in range(len(table[decider])) ] #return array of predictions

#creates the confusion matrix as a dictionary
def confusion_matrix(root,table,attributes):
    matrix = {}
    for i in attributes[decider]: #for values the results class can take
        for j in attributes[decider]: #for values the results class can take
            matrix[(i,j)] = 0 #create the confusion matrix/dictionary with initial values 0
    a = classify(root,table,attributes) #classify the data
    for i in range(len(a)): #for each result from classification
        matrix[(table[decider][i],a[i])] += 1 #increment the value at predicted vs original class

    printmatrix(matrix,attributes[decider]) #print the matrix

#prints the confusion matrix
def printmatrix(matrix,values):
    print("confussion matrix")
    print("",*values,sep="   ")
    sm = 0
    for i in values:
        if len(i) == 1:
            print("",i,end=" ")
        else:
            print(i,end=" ")
        for j in values:
            if i == j:
                sm += matrix[(i,j)]
            print(matrix[(i,j)],end=" ")
        print()
    print("\naccurecy =",str((sm/sum(list(matrix.values())))*100)+"%")

attributes,table,decider = readfile("Training Dataset.arff") #read the training data
root = ID3(table,attributes)

attributes1,table1 = readfile("old.arff")[:2] #read the validation data
table2 = {}
for i in table1: #split validation into validation and testing data
    n = len(table1[i])
    table2[i] = table1[i][int(n/2):n:]
    table1[i] = table1[i][:int(n/2):]

node = validate(root,table1,attributes1) #prune with validation data and set thr root of prunrd tree as node
confusion_matrix(node,table2,attributes1)
