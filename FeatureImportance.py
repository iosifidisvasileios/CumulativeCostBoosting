import warnings

from cycler import cycler

from DataPreprocessing.load_diabetes import load_diabetes
from DataPreprocessing.load_electricity import load_electricity
from DataPreprocessing.load_phoneme import load_phoneme
from DataPreprocessing.load_speed_dating import load_speed_dating

warnings.filterwarnings("ignore")
import os
from collections import defaultdict, Counter

import pickle
import matplotlib
import numpy
import operator

from multiprocessing import Process
from sklearn.metrics import f1_score, balanced_accuracy_score

from Competitors.RareBoost import RareBoost
from Competitors.CostBoostingAlgorithms import CostSensitiveAlgorithms
from DataPreprocessing.load_mat_data import load_mat_data
from AdaCC import AdaCC

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, 'DataPreprocessing')

import pandas as pd
from DataPreprocessing.load_adult import load_adult
from DataPreprocessing.load_wilt import load_wilt
from DataPreprocessing.load_mushroom import load_mushroom
from DataPreprocessing.load_eeg_eye import load_eeg_eye
from DataPreprocessing.load_spam import load_spam
from DataPreprocessing.load_skin import load_skin
from DataPreprocessing.load_credit import load_credit
from DataPreprocessing.load_kdd import load_kdd
from DataPreprocessing.load_bank import load_bank


def run_eval(dataset, base_learners, methods):
    if dataset == "wilt":
        X, y, cl_names = load_wilt()
    elif dataset == "adult":
        X, y, cl_names = load_adult()
    elif dataset == "diabetes":
        X, y, cl_names = load_diabetes()
    elif dataset == "phoneme":
        X, y, cl_names = load_phoneme()
    elif dataset == "mushroom":
        X, y, cl_names = load_mushroom()
    elif dataset == "electricity":
        X, y, cl_names = load_electricity()
    elif dataset == "speeddating":
        X, y, cl_names = load_speed_dating()
    elif dataset == "credit":
        X, y, cl_names = load_credit()
    elif dataset == "eeg_eye":
        X, y, cl_names = load_eeg_eye()
    elif dataset == "spam":
        X, y, cl_names = load_spam()
    elif dataset == "skin":
        X, y, cl_names = load_skin()
    elif dataset == "bank":
        X, y, cl_names = load_bank()
    elif dataset == "kdd":
        X, y, cl_names = load_kdd()
    elif dataset == "landsatM":
        X, y, cl_names = load_mat_data(dataset)
    elif dataset == "musk2":
        X, y, cl_names = load_mat_data(dataset)
    elif dataset == "spliceM":
        X, y, cl_names = load_mat_data(dataset)
    elif dataset == "semeion_orig":
        X, y, cl_names = load_mat_data(dataset)
    elif dataset == "waveformM":
        X, y, cl_names = load_mat_data(dataset)
    else:
        from imblearn import datasets

        data = datasets.fetch_datasets()[dataset]
        cl_names = ["feature_" + str(i) for i in range(0, data['data'].shape[1])]
        X = data['data']
        y = data['target']

    y[y != 1] = 0

    processes = []

    for method in methods:
        p = Process(target=train_classifier, args=(X, y, base_learners, method, cl_names))  # Passing the list
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    N = len(methods)
    ind = numpy.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    raw_data = dict()


    for method in methods:
        with open('temp_features/' + method, 'rb') as filehandle:
            # read the data as binary data stream
            model = pickle.load(filehandle)
            # print (method, model.feature_importances_)
            raw_data[method] = model.feature_importances_
            f_num = len(model.feature_importances_)
    index = ["Feature " + str(k) for k in range(1, f_num+1)]
    # index = ["Atrribute 1","Atrribute 2","Atrribute 3","Atrribute 4","Atrribute 5","Atrribute 6"]
    df = pd.DataFrame(raw_data, index=index)
    df = df.transpose()

    ax = df.plot.bar(stacked=True,alpha=0.75, rot=25)
    ax.set_ylabel("Feature importance")
    ax.set_xlabel("Methods")
    ax.legend(loc='center left', bbox_to_anchor=(0.1, 01.07), ncol=3)  # here is the magic

    ax.figure.savefig('Images/features/' + dataset +'.png',bbox_inches='tight', dpi=200)


def train_classifier(X_train, y_train, base_learners, method, cl_names):
    if method == 'AdaBoost':
        clf = CostSensitiveAlgorithms(algorithm='AdaBoost', n_estimators=base_learners)
        clf.fit(X_train, y_train)

    elif 'AdaCC' in method or 'AdaN-CC' in method:
        clf = AdaCC(n_estimators=base_learners, algorithm=method)
        clf.fit(X_train, y_train)

    elif 'RareBoost' in method:
        clf = RareBoost(n_estimators=base_learners)
        clf.fit(X_train, y_train)
    else:
        counter_dict = Counter(list(y_train))
        majority = max(counter_dict.items(), key=operator.itemgetter(1))[0]
        minority = max(counter_dict.items(), key=operator.itemgetter(0))[0]
        best = -1
        ratios = [1., 2., 3., 4., 5., 6., 7, 8., 9., 10.]

        for j in ratios:
            try:
                clf = CostSensitiveAlgorithms(n_estimators=base_learners, algorithm=method,
                                              class_weight={minority: 1, majority: j / 10.})
                clf.fit(X_train, y_train)
                if clf.error == 1:
                    clf = None
                else:
                    score = f1_score(y_train, clf.predict(X_train))
                    # score = balanced_accuracy_score(y_train, clf.predict(X_train))
                    if score >= best:
                        best = score
                        best_clf = clf
            except:
                pass
        clf = best_clf

    with open('temp_features/' + method, 'wb') as filehandle:
        pickle.dump(clf, filehandle)


if __name__ == '__main__':

    if not os.path.exists("temp_features"):
        os.makedirs("temp_features")

    list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'CGAda', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3', 'RareBoost']

    datasets_list = sorted(['adult', 'wilt', 'credit', 'spam', 'bank', 'musk2', 'isolet',
                            'abalone', 'car_eval_34', 'letter_img', 'protein_homo', 'skin', 'eeg_eye', 'phoneme',
                            'electricity', 'scene', 'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid',
                            'thyroid_sick','wine_quality', 'us_crime', 'ozone_level', 'webpage', 'coil_2000'])

    datasets_list = [ 'mammography', ]
    for baseL in [ 200]:
        overall_list = []
        for dataset in datasets_list:
            run_eval(dataset=dataset, base_learners=baseL, methods=list_of_methods)

