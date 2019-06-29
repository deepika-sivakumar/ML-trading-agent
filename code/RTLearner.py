# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:34:44 2019

@author: Deepika

"""

import numpy as np
from DTLearner import DTLearner # From module "DTLearner" (filename) import class "DTLearner"

class RTLearner(DTLearner):

    def __init__(self, leaf_size, verbose = False):
        DTLearner.__init__(self, leaf_size, verbose)

    def author(self):
        return 'Deepika'

    # To calculate the best feature random
    def findBestFeature(self, dataX, dataY):
        total_features = int(dataX.shape[1]) # Gives the no of features to select from
        features_array = np.arange(total_features) # Store the features in an array
        i = np.random.choice(features_array, 1, replace=True)
        return i[0] # Return only the element since i is an array
