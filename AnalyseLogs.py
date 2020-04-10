from DataPreprocessing.load_diabetes import load_diabetes
from DataPreprocessing.load_electricity import load_electricity
from DataPreprocessing.load_phoneme import load_phoneme
from DataPreprocessing.load_speed_dating import load_speed_dating
from DataPreprocessing.load_mat_data import load_mat_data
import sys
from imblearn import datasets
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


datasets_list = sorted(['mushroom', 'adult', 'wilt', 'credit', 'spam', 'bank', 'landsatM', 'musk2', 'isolet',
                        'spliceM', 'semeion_orig', 'waveformM', 'abalone', 'car_eval_34', 'letter_img',
                        'skin', 'eeg_eye', 'phoneme', 'electricity', 'scene',  # 'kdd' ,'diabetes',
                        'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
                        'wine_quality', 'us_crime', 'protein_homo', 'ozone_level', 'webpage', 'coil_2000'])

dataset_names = set()
list_of_stats = []
for dataset in datasets_list:
    list_of_stats.append(load_datasets(dataset, dataset_names))

logs = open("Evaluation.log", "r")

new_dataset = False
performance_flag = False
list_of_performance = []
dataset_performance = dict()
cnt = 0
for line in logs:
    if line.split("\t")[0] in dataset_names:
        performance_flag = False

    if line.startswith("balanced_accuracy"):
        performance_flag = True
        continue

    if performance_flag:
        tempList = line.replace("\n","").split(",")
        dataset_performance[tempList[0]] = float(tempList[-1].split(" ")[0])/100.
        if line.startswith('RareBoost'):
            list_of_performance.append(dataset_performance)
            dataset_performance = dict()
            performance_flag = False

opt_for = "feat"
if opt_for == "class":
    ranked_stats = [x[2] for x in list_of_stats]
elif opt_for == "feat":
    ranked_stats = [x[1] for x in list_of_stats]
elif opt_for == "inst":
    ranked_stats = [x[0] for x in list_of_stats]

indexes = list(numpy.argsort(ranked_stats))
ranked_x = list(numpy.sort(ranked_stats))
ranked_y = [list_of_performance[i] for i in indexes]


def plot_data(ranked_x, ranked_y, output_dir, filename):
    list_of_methods = ['AdaBoost', 'AdaAC1', 'AdaAC2', 'AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3','RareBoost']
    list_of_results = [[] for i in list_of_methods]

    for item in ranked_y:
        for idx, method in enumerate(list_of_methods):
            list_of_results[idx].append(item[method])

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 12})
    plt.grid()
    # plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
    plt.xscale('log')
    plt.grid(True, axis='y')

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']
    plt.ylabel('%')
    if filename == "class_based":
        plt.xlabel("Class Imbalance Ratio")
    elif filename == "feat_based":
        plt.xlabel("Number of Features")
    elif filename == "inst_based":
        plt.xlabel("Number of Instances")

    for i in range(0, len(list_of_methods)):
        plt.plot(ranked_x, list_of_results[i], label=list_of_methods[i], color=colors[i])

    plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.085), ncol=6)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + filename + ".png", bbox_inches='tight', dpi=200)
    plt.clf()


if opt_for == "class":
    plot_data(ranked_x, ranked_y, "Images/", "class_based")

elif opt_for == "feat":
    plot_data(ranked_x, ranked_y, "Images/", "feat_based")

elif opt_for == "inst":
    plot_data(ranked_x, ranked_y, "Images/", "inst_based")



