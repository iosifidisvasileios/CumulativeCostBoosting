from __future__ import division
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

import scipy.io as sio

SEED = 1234
seed(SEED)
np.random.seed(SEED)



def load_mat_data(dataset):
	mat_contents = sio.loadmat("DataPreprocessing/" + dataset + '.mat')
	data = mat_contents['data']
	target = np.asarray([float(i) for i in mat_contents['labels'].ravel()])
	target[np.where(target != 1)] = -1  # One-vs-all if multiclass
	cl_names = ["feature_" + str(i) for i in range(0, data.shape[1])]
	return data, target, cl_names
