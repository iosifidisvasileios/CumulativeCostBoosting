from __future__ import division
# import urllib2
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

# import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)



def load_mushroom():
	FEATURES_CLASSIFICATION = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment',
							   'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring',
							   'stalk-surface-below-ringm','stalk-color-above-ring','stalk-color-below-ring','veil-type',
							   'veil-color','ring-number','ring-type','spore-print-color','population','habitat'] #features to be used for classification
	CONT_VARIABLES = [] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
	CLASS_FEATURE = "class" # the decision variable


	# COMPAS_INPUT_FILE = "bank-full.csv"
	COMPAS_INPUT_FILE = "DataPreprocessing/mushroom.csv"


	# load the data and get some stats
	df = pd.read_csv(COMPAS_INPUT_FILE)

	# convert to np array
	data = df.to_dict('list')
	for k in data.keys():
		data[k] = np.array(data[k])


	""" Feature normalization and one hot encoding """
	# convert class label 0 to -1
	y = data[CLASS_FEATURE]
	y[y=='p'] = "1"
	y[y=="e"] = '0'
	y =  np.array([int(k) for k in y])
	y[y==0] = -1

	X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
	cl_names = []
	for attr in FEATURES_CLASSIFICATION:
		vals = data[attr]
		if attr in CONT_VARIABLES:
			vals = [float(v) for v in vals]
			vals = preprocessing.scale(vals) # 0 mean and 1 variance
			vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col
			cl_names.append(attr)
		else: # for binary categorical variables, the label binarizer uses just one var instead of two
			# lb = preprocessing.LabelBinarizer()
			# lb.fit(vals)
			# vals = lb.transform(vals)

			xxx =  pd.get_dummies(vals, prefix=attr, prefix_sep='?')

			cl_names += [at_in for at_in in xxx.columns]
			vals = xxx

		# add to learnable features
		X = np.hstack((X, vals))

	return X, y,cl_names