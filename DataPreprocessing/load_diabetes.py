from __future__ import division
# import urllib2
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

# sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
# import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)


def load_diabetes():
    FEATURES_CLASSIFICATION = ["race", "gender", "age", "weight", "admission_type_id",
                               "discharge_disposition_id", "admission_source_id", "time_in_hospital", "payer_code",
                               "medical_specialty", "num_lab_procedures", "num_procedures", "num_medications",
                               "number_outpatient", "number_emergency", "number_inpatient", "diag_1", "diag_2",
                               "diag_3", "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin", "repaglinide",
                               "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
                               "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
                               "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
                               "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
                               "metformin-pioglitazone", "change", "readmitted"]
    CONT_VARIABLES = ["admission_type_id",
                      "discharge_disposition_id", "admission_source_id", "time_in_hospital", "num_lab_procedures",
                      "num_procedures", "num_medications",
                      "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    CLASS_FEATURE = "diabetesMed"  # the decision variable

    COMPAS_INPUT_FILE = "DataPreprocessing/diabetic_data.csv"

    df = pd.read_csv(COMPAS_INPUT_FILE)

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y == 'No'] = "1"
    y[y == "Yes"] = '-1'
    y = np.array([int(k) for k in y])

    X = np.array([]).reshape(len(y), 0)  # empty array with num rows same as num examples, will hstack the features to it
    cl_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col
            cl_names.append(attr)
        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            xxx = pd.get_dummies(vals, prefix=attr, prefix_sep='?')
            cl_names += [at_in for at_in in xxx.columns]
            vals = xxx

        # add to learnable features
        X = np.hstack((X, vals))

    return X, y, cl_names
