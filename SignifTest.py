import pickle
import time
import warnings
from itertools import combinations

from Competitors.AdaMECal import AdaMEC
from DataPreprocessing.load_diabetes import load_diabetes
from DataPreprocessing.load_electricity import load_electricity
from DataPreprocessing.load_phoneme import load_phoneme
from DataPreprocessing.load_speed_dating import load_speed_dating

warnings.filterwarnings("ignore")


from Competitors.RareBoost import RareBoost
from DataPreprocessing.load_mat_data import load_mat_data
from scipy.stats import ttest_rel, ttest_ind, stats
from sklearn.model_selection import train_test_split

import joblib
from collections import defaultdict, Counter
import os, sys
import operator
from multiprocessing import Process

from imblearn import datasets
from sklearn.metrics import f1_score, balanced_accuracy_score
from AdaCC import AdaCC
from Competitors.AdaC1C3 import AdaCost
from DataPreprocessing.load_adult import load_adult
from DataPreprocessing.load_wilt import load_wilt
from DataPreprocessing.load_mushroom import load_mushroom
from DataPreprocessing.load_eeg_eye import load_eeg_eye
from DataPreprocessing.load_spam import load_spam
from DataPreprocessing.load_skin import load_skin
from DataPreprocessing.load_credit import load_credit
from DataPreprocessing.load_kdd import load_kdd
from DataPreprocessing.load_bank import load_bank

sys.path.insert(0, 'DataPreprocessing')


def print_stats(names, stats):
    for i in range(0, len(names)):
        print(names[i] + " " + str(stats[i]))


def run_eval(dataset, baseL, methods):
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
        data = datasets.fetch_datasets()[dataset]
        cl_names = ["feature_" + str(i) for i in range(0, data['data'].shape[1])]
        X = data['data']
        y = data['target']
    y[y != 1] = 0
    list_of_predictions = []

    for i in methods:
        list_of_predictions.append([])

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=int(time.time()))
        processes = []
        for method in methods:
            p = Process(target=train_and_predict, args=(X_train, y_train, X_test, baseL, method))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for index, method in enumerate(methods):
            with open('Sig_temp_preds/' + method, 'rb') as filehandle:
                list_of_predictions[index] += list(joblib.load(filehandle))

    return list_of_predictions


def train_and_predict(X_train, y_train, X_test, base_learners, method):
    if method == 'AdaBoost':
        clf = AdaCost(algorithm='AdaBoost', n_estimators=base_learners)
        clf.fit(X_train, y_train)
    elif 'AdaCC' in method:
        clf = AdaCC(n_estimators=base_learners, algorithm=method)
        clf.fit(X_train, y_train)

    elif 'AdaMEC' in method:
        counter_dict = Counter(list(y_train))

        majority = max(counter_dict.items(), key=operator.itemgetter(1))[0]
        minority = max(counter_dict.items(), key=operator.itemgetter(0))[0]
        ratios = [1., 2., 3., 4., 5., 6., 7, 8., 9., 10.]
        ratios = [2.,  4.,  6.,  8., 10.]

        clf = AdaMEC(n_estimators=base_learners, algorithm=method)
        clf.fit(X_train, y_train)
        best_score = -1
        best_idx = 0
        for idx, cost in enumerate(ratios):
            class_weight = {minority: 1, majority: cost / 10.}
            clf.set_costs(y_train, class_weight)
            score = f1_score(y_train, clf.predict(X_train))
            # score = balanced_accuracy_score(y_train, clf.predict(X_train))
            if best_score < score:
                best_idx = idx
                best_score = score
        class_weight = {minority: 1, majority: ratios[best_idx] / 10.}
        clf.set_costs(y_train, class_weight)

    elif 'RareBoost' in method:
        clf = RareBoost(n_estimators=base_learners)
        clf.fit(X_train, y_train)
    else:
        counter_dict = Counter(list(y_train))
        majority = max(counter_dict.items(), key=operator.itemgetter(1))[0]
        minority = max(counter_dict.items(), key=operator.itemgetter(0))[0]
        ratios = [1., 2., 3., 4., 5., 6., 7, 8., 9., 10.]
        ratios = [2.,  4.,  6.,  8., 10.]

        processes = []
        for ratio in ratios:
            p = Process(target=train_competitors,
                        args=(X_train, y_train, X_test, base_learners, method, majority, minority, ratio))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        best_ratio = -1

        for ratio in ratios:
            if os.path.exists('Sig_temp_preds/' + method + str(ratio)):
                with open('Sig_temp_preds/' + method + str(ratio), 'rb') as filehandle:
                    temp = pickle.load(filehandle)
                    if temp[0] > best_ratio:
                        best_ratio = temp[0]
                        preds = temp[1]

            if os.path.exists('Sig_temp_preds/' + method + str(ratio)):
                os.remove('Sig_temp_preds/' + method + str(ratio))

        with open('Sig_temp_preds/' + method, 'wb') as filehandle:
            joblib.dump(preds, filehandle)
        return

    with open('Sig_temp_preds/' + method, 'wb') as filehandle:
        joblib.dump(clf.predict(X_test), filehandle)


def train_competitors(X_train, y_train, X_test, base_learners, method, maj, min, ratio):
    try:
        out = []
        clf = AdaCost(n_estimators=base_learners, algorithm=method, class_weight={min: 1, maj: ratio / 10.})
        clf.fit(X_train, y_train)
        out.append(f1_score(y_train, clf.predict(X_train)))
        # out.append(balanced_accuracy_score(y_train, clf.predict(X_train)))
        out.append(clf.predict(X_test))
        with open('Sig_temp_preds/' + method + str(ratio), 'wb') as filehandle:
            pickle.dump(out, filehandle)
    except:
        return

def sig_scores(dataset_predictions, baseL, methods):
    my_results = [defaultdict(int), defaultdict(int)]

    for dataset in dataset_predictions:

        others = []
        names_others = []
        names_mymethods = []
        mymethods = []

        for i in range(0, len(dataset)):
            if 'AdaCC' in methods[i]:
                mymethods.append(dataset[i])
                names_mymethods.append(methods[i])
                continue
            others.append(dataset[i])
            names_others.append(methods[i])

        alpha = 0.05

        for idx in range(0, len(others)):

            for k in range(0, len(my_results)):

                _, pvalue = stats.ttest_ind(mymethods[k], others[idx])
                if pvalue <= alpha:
                    my_results[k][names_others[idx]] += 1

        perm = combinations([0, 1], 2)
        for i in list(perm):
            _, pvalue = stats.ttest_ind(mymethods[i[0]], mymethods[i[1]])
            if pvalue <= alpha:
                my_results[i[0]][names_mymethods[i[1]]] += 1
                my_results[i[1]][names_mymethods[i[0]]] += 1

    for i in range(0, len(names_mymethods)):
        print("==============================", names_mymethods[i], str(baseL) + "============================")
        print(my_results[i])


if __name__ == '__main__':
    if not os.path.exists("Sig_temp_preds"):
        os.makedirs("Sig_temp_preds")
    baselines = [25, 50, 75, 100, 125, 150, 175, 200]
    list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2','AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3', 'RareBoost']

    # datasets_list = sorted(['mushroom', 'adult', 'wilt', 'credit', 'spam', 'bank', 'landsatM', 'musk2', 'isolet',
    #                         'spliceM', 'semeion_orig', 'waveformM', 'abalone', 'car_eval_34', 'letter_img',
    #                         'skin', 'eeg_eye', 'phoneme', 'electricity', 'scene',  # 'kdd' ,'diabetes',
    #                         'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
    #                         'wine_quality', 'us_crime', 'protein_homo', 'ozone_level', 'webpage', 'coil_2000'])

    datasets_list = sorted(['adult', 'wilt', 'credit', 'spam', 'bank', 'musk2', 'isolet',
                            'abalone', 'car_eval_34', 'letter_img', 'protein_homo', 'skin', 'eeg_eye', 'phoneme', 'electricity',
                            'scene', 'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
                            'wine_quality', 'us_crime', 'ozone_level', 'webpage', 'coil_2000'])


    for baseL in baselines:
        dataset_predictions = []

        for dataset in datasets_list:
            print(dataset, baseL)
            dataset_predictions.append(run_eval(dataset=dataset, baseL=baseL, methods=list_of_methods))

        sig_scores(dataset_predictions, baseL, list_of_methods)
