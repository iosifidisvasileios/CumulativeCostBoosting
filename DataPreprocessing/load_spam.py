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



def load_spam():
	FEATURES_CLASSIFICATION =  ['word_freq_make','word_freq_address','word_freq_all',
								'word_freq_3d','word_freq_our','word_freq_over',
								'word_freq_remove','word_freq_internet','word_freq_order',
								'word_freq_mail','word_freq_receive','word_freq_will',
								'word_freq_people','word_freq_report','word_freq_addresses',
								'word_freq_free','word_freq_business','word_freq_email',
								'word_freq_you','word_freq_credit','word_freq_your',
								'word_freq_font','word_freq_000','word_freq_money','word_freq_hp',
								'word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab',
								'word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data',
								'word_freq_415','word_freq_85','word_freq_technology','word_freq_1999',
								'word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs',
								'word_freq_meeting','word_freq_original','word_freq_project',
								'word_freq_re','word_freq_edu','word_freq_table',
								'word_freq_conference','char_freq_;','char_freq_(',
								'char_freq_[','char_freq_!','char_freq_$']


	CONT_VARIABLES = ['word_freq_make','word_freq_address','word_freq_all',
								'word_freq_3d','word_freq_our','word_freq_over',
								'word_freq_remove','word_freq_internet','word_freq_order',
								'word_freq_mail','word_freq_receive','word_freq_will',
								'word_freq_people','word_freq_report','word_freq_addresses',
								'word_freq_free','word_freq_business','word_freq_email',
								'word_freq_you','word_freq_credit','word_freq_your',
								'word_freq_font','word_freq_000','word_freq_money','word_freq_hp',
								'word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab',
								'word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data',
								'word_freq_415','word_freq_85','word_freq_technology','word_freq_1999',
								'word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs',
								'word_freq_meeting','word_freq_original','word_freq_project',
								'word_freq_re','word_freq_edu','word_freq_table',
								'word_freq_conference','char_freq_;','char_freq_(',
								'char_freq_[','char_freq_!','char_freq_$']


	CLASS_FEATURE = "class" # the decision variable


	# COMPAS_INPUT_FILE = "bank-full.csv"
	COMPAS_INPUT_FILE = "DataPreprocessing/spam.csv"


	# load the data and get some stats
	df = pd.read_csv(COMPAS_INPUT_FILE)

	# convert to np array
	data = df.to_dict('list')
	for k in data.keys():
		data[k] = np.array(data[k])


	""" Feature normalization and one hot encoding """
	# convert class label 0 to -1
	y = data[CLASS_FEATURE]
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