from __future__ import division
# import urllib2
import os, sys
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


def load_speed_dating():
    FEATURES_CLASSIFICATION = ["has_null", "wave", "gender", "age", "age_o", "d_age", "d_d_age", "race", "race_o",
                               "samerace", "importance_same_race", "importance_same_religion", "d_importance_same_race",
                               "d_importance_same_religion", "field", "pref_o_attractive", "pref_o_sincere",
                               "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests",
                               "d_pref_o_attractive", "d_pref_o_sincere", "d_pref_o_intelligence", "d_pref_o_funny",
                               "d_pref_o_ambitious", "d_pref_o_shared_interests", "attractive_o", "sinsere_o",
                               "intelligence_o", "funny_o", "ambitous_o", "shared_interests_o", "d_attractive_o",
                               "d_sinsere_o", "d_intelligence_o", "d_funny_o", "d_ambitous_o", "d_shared_interests_o",
                               "attractive_important", "sincere_important", "intellicence_important", "funny_important",
                               "ambtition_important", "shared_interests_important", "d_attractive_important",
                               "d_sincere_important", "d_intellicence_important", "d_funny_important",
                               "d_ambtition_important", "d_shared_interests_important", "attractive", "sincere",
                               "intelligence", "funny", "ambition", "d_attractive", "d_sincere", "d_intelligence",
                               "d_funny", "d_ambition", "attractive_partner", "sincere_partner", "intelligence_partner",
                               "funny_partner", "ambition_partner", "shared_interests_partner", "d_attractive_partner",
                               "d_sincere_partner", "d_intelligence_partner", "d_funny_partner", "d_ambition_partner",
                               "d_shared_interests_partner", "sports", "tvsports", "exercise", "dining", "museums",
                               "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
                               "music", "shopping", "yoga", "d_sports", "d_tvsports", "d_exercise", "d_dining",
                               "d_museums", "d_art", "d_hiking", "d_gaming", "d_clubbing", "d_reading", "d_tv",
                               "d_theater", "d_movies", "d_concerts", "d_music", "d_shopping", "d_yoga",
                               "interests_correlate", "d_interests_correlate", "expected_happy_with_sd_people",
                               "expected_num_interested_in_me", "expected_num_matches",
                               "d_expected_happy_with_sd_people", "d_expected_num_interested_in_me",
                               "d_expected_num_matches", "like", "guess_prob_liked", "d_like", "d_guess_prob_liked",
                               "met", "decision", "decision_o"]  # features to be used for classification
    CONT_VARIABLES = ["has_null", "wave","age", "age_o", "d_age","samerace", "importance_same_race", "importance_same_religion", "pref_o_attractive", "pref_o_sincere",
                               "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests","attractive_o", "sinsere_o",
                               "intelligence_o", "funny_o", "ambitous_o", "shared_interests_o", "attractive_important", "sincere_important", "intellicence_important", "funny_important",
                               "ambtition_important", "shared_interests_important","attractive", "sincere",
                               "intelligence", "funny", "ambition",  "attractive_partner", "sincere_partner", "intelligence_partner",
                               "funny_partner", "ambition_partner", "shared_interests_partner",  "sports", "tvsports", "exercise", "dining", "museums",
                               "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
                               "music", "shopping", "yoga", "interests_correlate", "expected_happy_with_sd_people",
                               "expected_num_interested_in_me", "expected_num_matches","like", "guess_prob_liked", "met", "decision", "decision_o"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "match"  # the decision variable

    # COMPAS_INPUT_FILE = "bank-full.csv"
    COMPAS_INPUT_FILE = "DataPreprocessing/speeddating.csv"

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Feature normalization and one hot encoding """
    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y == 0] = -1

    X = np.array([]).reshape(len(y),
                             0)  # empty array with num rows same as num examples, will hstack the features to it
    cl_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col
            cl_names.append(attr)
        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            # lb = preprocessing.LabelBinarizer()
            # lb.fit(vals)
            # vals = lb.transform(vals)

            xxx = pd.get_dummies(vals, prefix=attr, prefix_sep='?')

            cl_names += [at_in for at_in in xxx.columns]
            vals = xxx

        # add to learnable features
        X = np.hstack((X, vals))
    print (X.shape)
    print (X[~np.isinf(X).any(axis=1)].shape)
    return X, y, cl_names
