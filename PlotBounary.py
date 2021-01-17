import pickle
import time
import warnings
import matplotlib
from imblearn.datasets import make_imbalance
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_blobs, make_circles, make_classification
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, SparsePCA

from Competitors.AdaMEC_Cal import AdaMEC_Cal
from DataPreprocessing.load_diabetes import load_diabetes
from DataPreprocessing.load_electricity import load_electricity
from DataPreprocessing.load_phoneme import load_phoneme
from DataPreprocessing.load_speed_dating import load_speed_dating

warnings.filterwarnings("ignore")

from Competitors.RareBoost import RareBoost
from DataPreprocessing.load_mat_data import load_mat_data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import joblib
from plot_functions import plot_decision_boundary, retrieve_n_class_color_cubic
from collections import defaultdict, Counter
import os, sys
import operator
from multiprocessing import Process

from imblearn import datasets
from sklearn.metrics import f1_score, balanced_accuracy_score
from AdaCC import AdaCC
from Competitors.CostBoostingAlgorithms import CostSensitiveAlgorithms
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


def get_dataset(dataset):
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
    elif dataset not in ['bloob', 'circle', 'moon']:
        from imblearn import datasets

        data = datasets.fetch_datasets()[dataset]
        cl_names = ["feature_" + str(i) for i in range(0, data['data'].shape[1])]
        X = data['data']
        y = data['target']
    y[y != 1] = 0

    return X, y, cl_names


def run_eval(dataset, base_list, methods):
    data_list_of_predictions = []
    datasets_list = []
    if dataset == 'moon':
        X, y = make_moons(n_samples=1000, noise=0.25, shuffle=True)
        X, y = make_imbalance(X, y, sampling_strategy={0: 500, 1: 375}, random_state=1)
        X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        datasets_list.append((X, y, x_test, y_test))
        X, y = make_moons(n_samples=1000, noise=0.25, shuffle=True)
        X, y = make_imbalance(X, y, sampling_strategy={0: 500, 1: 250}, random_state=1)
        X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        datasets_list.append((X, y, x_test, y_test))
        X, y = make_moons(n_samples=1000, noise=0.25, shuffle=True)
        X, y = make_imbalance(X, y, sampling_strategy={0: 500, 1: 125}, random_state=1)
        X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        datasets_list.append((X, y, x_test, y_test))
    elif dataset == 'circle':
        X, y = make_circles(n_samples=1000, noise=0.25, factor=.7, shuffle=True)
        X, y = make_imbalance(X, y, sampling_strategy={0: 500, 1: 375}, random_state=1)
        X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        datasets_list.append((X, y, x_test, y_test))
        X, y = make_circles(n_samples=1000, noise=0.25, factor=.7, shuffle=True)
        X, y = make_imbalance(X, y, sampling_strategy={0: 500, 1: 250}, random_state=1)
        X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        datasets_list.append((X, y, x_test, y_test))
        X, y = make_circles(n_samples=1000, noise=0.25, factor=.7, shuffle=True)
        X, y = make_imbalance(X, y, sampling_strategy={0: 500, 1: 125}, random_state=1)
        X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        datasets_list.append((X, y, x_test, y_test))

    elif dataset == 'bloob':
        X, y = make_blobs(n_samples=1000, centers=2, cluster_std=4.5, random_state=1)
        X, y = make_imbalance(X, y, sampling_strategy={0: 500, 1: 375}, random_state=1)
        X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        datasets_list.append((X, y, x_test, y_test))
        X, y = make_blobs(n_samples=1000, centers=2, cluster_std=4.5, random_state=1)
        X, y = make_imbalance(X, y, sampling_strategy={0: 500, 1: 250}, random_state=1)
        X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        datasets_list.append((X, y, x_test, y_test))
        X, y = make_blobs(n_samples=1000, centers=2, cluster_std=4.5, random_state=1)
        X, y = make_imbalance(X, y, sampling_strategy={0: 500, 1: 125}, random_state=1)
        X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        datasets_list.append((X, y, x_test, y_test))
    else:
        X_orig, y_orig, _ = get_dataset(dataset)
        # X_orig = SparsePCA(n_components=2).fit_transform(X_orig)
        X_orig = PCA(n_components=2).fit_transform(X_orig)
        X, x_test, y, y_test = train_test_split(X_orig, y_orig, test_size=0.25, stratify=y_orig, random_state=17)
        datasets_list = [X, y, x_test, y_test]

    for baseL in base_list:
        list_of_predictors = []
        processes = []
        for method in methods:
            p = Process(target=train_and_predict, args=(datasets_list[0], datasets_list[1], baseL, method))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for index, method in enumerate(methods):
            with open('boundary_temp_preds/' + method, 'rb') as filehandle:
                list_of_predictors.append(joblib.load(filehandle))
        data_list_of_predictions.append(list_of_predictors)

    figure = plt.figure(figsize=(27, 12))
    plt.rcParams.update({'font.size': 16})

    i = 1

    for ds_cnt, ds in enumerate(base_list):
        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#00FF00'])
        ax = plt.subplot(len(base_list), len(data_list_of_predictions[ds_cnt]) + 2, i)

        if ds_cnt == 0:
            ax.set_title("Training data")

        x_min, x_max = datasets_list[0][:, 0].min() - 1, datasets_list[0][:, 0].max() + 1
        y_min, y_max = datasets_list[0][:, 1].min() - 1, datasets_list[0][:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        ax.scatter(datasets_list[0][:, 0], datasets_list[0][:, 1], c=datasets_list[1], cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        if ds_cnt == 0:
            ax.set_ylabel('25 weak learners')
        elif ds_cnt == 1:
            ax.set_ylabel('50 weak learners')
        elif ds_cnt == 2:
            ax.set_ylabel('100 weak learners')
        elif ds_cnt == 3:
            ax.set_ylabel('200 weak learners')

        i += 1
        ax = plt.subplot(len(base_list), len(data_list_of_predictions[ds_cnt]) + 2, i)

        if ds_cnt == 0:
            ax.set_title("Testing data")

        # x_min, x_max = datasets_list[2][:, 0].min() - 1, datasets_list[2][:, 0].max() + 1
        # y_min, y_max = datasets_list[2][:, 1].min() - 1, datasets_list[2][:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        ax.scatter(datasets_list[2][:, 0], datasets_list[2][:, 1], c=datasets_list[3], cmap=cm_bright, alpha=1,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        i += 1

        for name, clf in zip(methods, data_list_of_predictions[ds_cnt]):
            score = balanced_accuracy_score(datasets_list[3], clf.predict(datasets_list[2]))
            print(name, score)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax = plt.subplot(len(base_list), len(data_list_of_predictions[ds_cnt]) + 2, i)
            ax.contourf(xx, yy, Z, cmap=cm_bright, alpha=.9)
            # ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')
            # Plot the testing points
            ax.scatter(datasets_list[2][:, 0], datasets_list[2][:, 1], c=datasets_list[3], cmap=cm_bright,
                       edgecolors='k', alpha=.9)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.3f' % score).lstrip('0'), size=20, weight='bold', horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.show()
    plt.savefig("Images/Boundaries/" + dataset + ".png", bbox_inches='tight', dpi=200)


def train_and_predict(X_train, y_train, base_learners, method):
    if method == 'AdaBoost':
        clf = CostSensitiveAlgorithms(algorithm='AdaBoost', n_estimators=base_learners)
        clf.fit(X_train, y_train)
    elif 'AdaCC' in method:
        clf = AdaCC(n_estimators=base_learners, algorithm=method)
        clf.fit(X_train, y_train)

    elif 'AdaMEC' in method:
        counter_dict = Counter(list(y_train))

        majority = max(counter_dict.items(), key=operator.itemgetter(1))[0]
        minority = max(counter_dict.items(), key=operator.itemgetter(0))[0]
        ratios = [1., 2., 3., 4., 5., 6., 7, 8., 9., 10.]

        clf = AdaMEC_Cal(n_estimators=base_learners, algorithm=method)
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
        ratios = [1., 2., 3., 4., 5., 6., 7, 8., 9., 9.99]

        processes = []
        for ratio in ratios:
            p = Process(target=train_competitors,
                        args=(X_train, y_train, base_learners, method, majority, minority, ratio))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        best_ratio = -1

        predictor = None
        for ratio in ratios:
            if os.path.exists('boundary_temp_preds/' + method + str(ratio)):
                with open('boundary_temp_preds/' + method + str(ratio), 'rb') as filehandle:
                    temp = pickle.load(filehandle)
                    if temp[0] > best_ratio:
                        best_ratio = temp[0]
                        predictor = temp[1]

            if os.path.exists('boundary_temp_preds/' + method + str(ratio)):
                os.remove('boundary_temp_preds/' + method + str(ratio))

        with open('boundary_temp_preds/' + method, 'wb') as filehandle:
            joblib.dump(predictor, filehandle)
        return

    with open('boundary_temp_preds/' + method, 'wb') as filehandle:
        joblib.dump(clf, filehandle)


def train_competitors(X_train, y_train, base_learners, method, maj, min, ratio):
    try:
        out = []
        clf = CostSensitiveAlgorithms(n_estimators=base_learners, algorithm=method, class_weight={min: 1, maj: ratio / 10.})
        clf.fit(X_train, y_train)
        out.append(f1_score(y_train, clf.predict(X_train)))
        # out.append(balanced_accuracy_score(y_train, clf.predict(X_train)))
        out.append(clf)
        with open('boundary_temp_preds/' + method + str(ratio), 'wb') as filehandle:
            pickle.dump(out, filehandle)
    except:
        return


if __name__ == '__main__':
    if not os.path.exists("boundary_temp_preds"):
        os.makedirs("boundary_temp_preds")
    if not os.path.exists("Images/Boundaries"):
        os.makedirs("Images/Boundaries")
    list_of_methods = ['AdaBoost', 'AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3',
                       'RareBoost', 'AdaCC1', 'AdaCC2']

    datasets_list = sorted(['adult', 'wilt', 'credit', 'spam', 'bank', 'musk2', 'isolet',
                            'abalone', 'car_eval_34', 'letter_img', 'protein_homo', 'skin', 'eeg_eye', 'phoneme', 'electricity',
                            'scene', 'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
                            'wine_quality', 'us_crime', 'ozone_level', 'webpage', 'coil_2000'])

    # run_eval(dataset=['adult', 'optical_digits', 'musk2' ], baseL=200, methods=list_of_methods)
    # for data in datasets_list:
    #     print (data)
    #     run_eval(dataset=data, base_list=[25, 50, 100, 200], methods=list_of_methods)
    # run_eval(dataset='wine_quality', base_list=[1 ], methods=list_of_methods)
    run_eval(dataset='wine_quality', base_list=[25, 50, 100, 200], methods=list_of_methods)
