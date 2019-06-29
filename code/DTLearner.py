# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:34:44 2019

@author: Deepika

"""

import numpy as np
from scipy import stats

class DTLearner(object):

    def __init__(self, leaf_size, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.decision_tree = np.empty((0,4), float) #Use this to create the tree
        self.predictY = np.array([])
        self.test_node_val = 0

    def author(self):
        return 'Deepika'

    def findBestFeature(self, dataX, dataY):
        # Calculate the correlation values between each factor in dataX & dataY
        corr = []
        for column in np.transpose(dataX):
            c = np.corrcoef(column, dataY)
            corr.append(abs(c[0,1]))
        # Find the index of the factor with the highest absolute correlation
        i = np.nanargmax(corr)
        return i

    def buildTree(self,data):
        # Slice data into dataX and dataY for computing
        dataX = data[:,:-1] # Select all the rows, but except the last column
        dataY = data[:,[-1]] # Select all the rows of only the last column
        dataY = np.transpose(dataY)
        # Set distinct values for the leaf node
        leaf = -1
        NA = -1

        # Handle scenrios when data has only one row or all Y values are same
        if(int(data.shape[0]) == 1 or len(np.unique(dataY)) == 1):
            leaf_array = np.array([[leaf, dataY[0,0], NA, NA]])
            return leaf_array
        # Handle scenario when the sample size <= leaf_size
        if(int(data.shape[0]) <= self.leaf_size):
            # Compute mode instead of mean for classification problem
            mode = stats.mode(dataY)
            leaf_array = np.array([[leaf, mode[0][0][0], NA, NA]])
#            leaf_array = np.array([[leaf, np.mean(dataY), NA, NA]])
            return leaf_array

        # Else build the decision tree
        # Call the function to find the best feature
        i = self.findBestFeature(dataX, dataY)
        # Find the split value by calculating the mean of the values of the ith factor
        split_val = np.median(dataX[:,i])
        # Sort the data by the chosen factor
        data = data[data[:,i].argsort()]

        # Calculate the value for the left node - left node always goes one depth down the root
        left_node = 1
        # Spilt the data using the split value to construct the left tree
        left_tree_data = data[data[:,i] <= split_val]
        if(int(left_tree_data.shape[0]) == int(data.shape[0])):
            # Compute mode instead of mean for classification problem
            mode = stats.mode(dataY)
            leaf_array = np.array([[leaf, mode[0][0][0], NA, NA]])
#            leaf_array = np.array([[leaf, np.mean(dataY), NA, NA]])
            return leaf_array
        # Build the left tree
        left_tree = self.buildTree(left_tree_data)
        # Calculate the right_node value - right node always goes one + left tree's shape down the root
        right_node = int(left_tree.shape[0] + 1)
        # Split the data using the split value to construct the right tree
        right_tree = self.buildTree(data[data[:,i] > split_val])
        # Construct the root node
        root = np.array([i, split_val, left_node, right_node])
        return np.vstack((root, left_tree, right_tree))

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # Concatenate the two arrays
        data = np.column_stack((dataX,dataY))
        # Build the tree
        self.decision_tree = self.buildTree(data)
        return

    def singleQuery(self,singleTestX):
        """
        @summary: Search the decision tree for each row of dataX and append the value to self.predictY
        @param singleTestX: Single row of dataX values
        """
        leaf = -1
        # Select the row in the decision tree to test the data
        decision_node = self.decision_tree[self.test_node_val]

        # If the decision node is the leaf node, then take the value from that as the predicted Y
        if(decision_node[0] == leaf):
            self.predictY = np.append(self.predictY, decision_node[1])
            return

        # If its not a leaf node, then get the 'Vector' value to test with
        decision_vector = int(decision_node[0])

        # If the vector th value is less than or equal to the split value, traverse down the left node
        # else traverse down the right node
        if(singleTestX[decision_vector] <= decision_node[1]):
            # If we have to traverse down the left tree, it is just the current node value + its left node value(which is always 1)
            self.test_node_val = int(self.test_node_val + decision_node[2])
            self.singleQuery(singleTestX)
        else:
            # If we have to traverse down the right tree, it is the current node value + its right node value
            self.test_node_val = int(self.test_node_val + decision_node[3])
            self.singleQuery(singleTestX)
        return

    def query(self,testX):
        self.predictY = np.array([])
        total_test_rows = int(testX.shape[0])
        if self.verbose == True:
            print('total_test_rows::',total_test_rows)
        # Loop through all the X samples and get the predicted Y values
        for i in range(0,total_test_rows):
            # Reset the test node value to 0, so for each dataset the query traverses from the top of the tree
            self.test_node_val = 0
            self.singleQuery(testX[i])
        return self.predictY

if __name__=="__main__":
    print "Decision Tree Learner"
