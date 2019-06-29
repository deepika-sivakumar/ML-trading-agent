# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 08:39:30 2019

@author: Deepika

"""

import numpy as np
#import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
from scipy import stats

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learnerEnsemble = []
        # Create the appropriate learner instances for the no. of bags
        for i in range(0,self.bags):
            self.learnerEnsemble.append(self.learner(**self.kwargs))

    def author(self):
        return 'Deepika'

    def addEvidence(self,dataX,dataY):
        # Concatenate the two arrays
        data = np.column_stack((dataX,dataY))

        total_train_rows = int(data.shape[0])
        total_train_cols = int(data.shape[1])
        n_array = np.arange(total_train_rows)

        # Add evidence to all the learner instances / bags
        for j in range(len(self.learnerEnsemble)):
            # Get the random n rows to choose from the dataset for each bag
            random_n_rows = np.random.choice(n_array, total_train_rows, replace=True)
            randomData = np.empty((0, total_train_cols), float)
            for k in range(int(random_n_rows.shape[0])):
                randomData = np.vstack((randomData, data[random_n_rows[k]]))
            # Now split the data to pass to the addEvidence method of each learner instance
            randomDataX = randomData[:,:-1] # Select all the rows, but except the last column
            randomDataY = randomData[:,-1] # Select all the rows of only the last column
            self.learnerEnsemble[j].addEvidence(randomDataX, randomDataY)
        return

    def query(self,testX):
        predictY = np.empty((0, int(testX.shape[0])), float)
        for i in range(self.bags):
            predictY = np.vstack((predictY, self.learnerEnsemble[i].query(testX)))
        # Compute mode instead of mean for classification problem
        mode = stats.mode(predictY, axis=0)
        Y = mode[0][0]
#        Y = np.mean(predictY, axis=0)
        return Y

if __name__=="__main__":
    print "Bag Learner"
