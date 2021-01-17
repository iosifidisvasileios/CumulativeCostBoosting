from collections import defaultdict

from DataPreprocessing.load_diabetes import load_diabetes
from DataPreprocessing.load_electricity import load_electricity
from DataPreprocessing.load_phoneme import load_phoneme
from DataPreprocessing.load_speed_dating import load_speed_dating
from DataPreprocessing.load_mat_data import load_mat_data
import sys
from DataPreprocessing.load_adult import load_adult
from DataPreprocessing.load_wilt import load_wilt
from DataPreprocessing.load_mushroom import load_mushroom
from DataPreprocessing.load_eeg_eye import load_eeg_eye
from DataPreprocessing.load_spam import load_spam
from DataPreprocessing.load_skin import load_skin
from DataPreprocessing.load_credit import load_credit
from DataPreprocessing.load_kdd import load_kdd
from DataPreprocessing.load_bank import load_bank
# encoding: utf-8

import matplotlib
import os

matplotlib.use('Agg')

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, 'DataPreprocessing')


def load_datasets(dataset, names):
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

    names.add(dataset)
    output = []
    output.append(X.shape[0])
    output.append(X.shape[1])
    output.append(float(format(len(abs(y[y != 1])) / sum(y[y == 1]), '.2f')))

    return output


datasets_list = sorted(['adult', 'wilt', 'credit', 'spam', 'bank', 'musk2', 'isolet',
                        'abalone', 'car_eval_34', 'letter_img', 'protein_homo', 'skin', 'eeg_eye', 'phoneme',
                        'electricity',
                        'scene', 'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid',
                        'thyroid_sick',
                        'wine_quality', 'us_crime', 'ozone_level', 'webpage', 'coil_2000'])

dataset_names = set()
list_of_stats = []
for dataset in datasets_list:
    list_of_stats.append(load_datasets(dataset, dataset_names))

accuracy = []
gmean = []
f1score = []
recall = []
balanced_accuracy = []
opm = []

list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3',
                   'RareBoost']
measures = ['accuracy', 'gmean', 'f1score', 'recall', 'balanced_accuracy', 'opm']

for baseL_index in range(1, 9):
    for item in measures:
        logs = open("eval.log", "r")
        new_dataset = False
        performance_flag = False
        list_of_performance = []
        dataset_performance = dict()
        cnt = 0
        for line in logs:
            if line.split("\t")[0] in dataset_names:
                performance_flag = False
                continue

            if line.startswith(item):
                performance_flag = True
                continue

            if performance_flag:
                tempList = line.replace("\n", "").split(",")
                if tempList[0] in list_of_methods:
                    dataset_performance[tempList[0]] = float(tempList[baseL_index].split(" ")[0]) / 100.
                    if line.startswith('RareBoost'):
                        list_of_performance.append(dataset_performance)
                        dataset_performance = dict()
                        performance_flag = False

        logs.close()

        ranked_stats = defaultdict(int)
        for dict_measure in list_of_performance:
            list_ranked = sorted(dict_measure.items(), key=lambda x: x[1], reverse=True)
            for idx_rank, ranking in enumerate(list_ranked):
                ranked_stats[ranking[0]] += (idx_rank + 1) / float(len(list_of_stats))

        if item == 'accuracy':
            accuracy.append(ranked_stats)
        elif item == 'gmean':
            gmean.append(ranked_stats)
        elif item == 'f1score':
            f1score.append(ranked_stats)
        elif item == 'recall':
            recall.append(ranked_stats)
        elif item == 'balanced_accuracy':
            balanced_accuracy.append(ranked_stats)
        elif item == 'opm':
            opm.append(ranked_stats)

        # print(item, baseL_index, ranked_stats)


def plot_data(list_of_measures, output_dir, filename):
    list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3',
                       'RareBoost']
    list_of_results = [[] for i in list_of_methods]

    for dict_temp in list_of_measures:
        for idx, method in enumerate(list_of_methods):
            list_of_results[idx].append(dict_temp[method])

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 12})
    plt.grid()
    # plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']
    plt.ylabel('Rank')

    plt.xlabel("Weak learners")

    x_ticks = [25, 50, 75, 100, 125, 150, 175, 200]
    for i in range(0, len(list_of_methods)):
        plt.plot(x_ticks, list_of_results[i], label=list_of_methods[i], color=colors[i])

    plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.085), ncol=6)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + filename + ".png", bbox_inches='tight', dpi=200)
    plt.clf()


plot_data(accuracy, 'Images/Ranking/', 'rank_accuracy')
plot_data(gmean, 'Images/Ranking/', 'rank_gmean')
plot_data(f1score, 'Images/Ranking/', 'rank_f1score')
plot_data(recall, 'Images/Ranking/', 'rank_recall')
plot_data(balanced_accuracy, 'Images/Ranking/', 'rank_balanced_accuracy')
plot_data(opm, 'Images/Ranking/', 'rank_opm')

#
# if opt_for == "class":
#     plot_data(ranked_x, ranked_y, "Images/", "class_based")
#
# elif opt_for == "feat":
#     plot_data(ranked_x, ranked_y, "Images/", "feat_based")
#
# elif opt_for == "inst":
#     plot_data(ranked_x, ranked_y, "Images/", "inst_based")
#
#
#
