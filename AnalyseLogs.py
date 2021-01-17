from operator import itemgetter

from cycler import cycler

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



def plot_data(ranked_x, ranked_y, output_dir, filename):
    list_of_methods = ['AdaCC1', 'AdaCC2', 'AdaBoost', 'AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3','RareBoost']
    list_of_results = [[] for i in list_of_methods]
    for i in range(0,len(ranked_x)):
        ranked_x[i] = ranked_x[i].replace("_", " ")
        ranked_x[i] = ranked_x[i].replace(" 34", "")

    for item in ranked_y:
        for idx, method in enumerate(list_of_methods):
            list_of_results[idx].append(item[method])

    plt.figure(figsize=(25, 5))
    plt.rcParams.update({'font.size': 15.5})
    colors = [ '#1f77b4', '#ff7f0e','#2ca02c', '#d62728', '#9467bd', '#8c564b',
               '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black']
    default_cycler = (cycler(color=colors)
                      # + cycler(linestyle=['-', (0, (1, 1)), '--', '-.',
                      #                   (0, (5, 10)),
                      #                   (0, (5, 1)),
                      #                   '-', (0, (1, 1)), '--', '-.',
                      #                   (0, (5, 10))])
                      +cycler(marker=[ 'd','v', 'x', '*', 'p', 'X', '^', 's', 'p', 'h', '8']))
    plt.rc('axes', prop_cycle=default_cycler)
    plt.grid()
    plt.grid(True, axis='y')

    x = [jj for jj in range(0,len(ranked_x))]
    # plt.xticks(x, ranked_x)
    plt.xlim([-.5,26.5])
    zord = [2,1,0,0,0,0,0,0,0,0,0,0]
    markers = [ 'd','v', 'x', '*', 'p', 'X', '^', 's', 'p', 'h', '8']
    for i in range(0, len(list_of_methods)):
        # plt.plot(x, list_of_results[i], label=list_of_methods[i], markersize=12.5)
        if i == 0 or i == 1:
            s=130
        else:
            s=90
        plt.scatter(x, list_of_results[i], label=list_of_methods[i], marker=markers[i], s=s,zorder= zord[i] )
    plt.xticks(x, ranked_x, rotation=20,ha='right')

    plt.legend(loc='upper center', bbox_to_anchor=(0.486, 1.155), ncol=11)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + filename + ".png", bbox_inches='tight', dpi=200)
    plt.clf()

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
                        'abalone', 'car_eval_34', 'letter_img', 'protein_homo', 'skin', 'eeg_eye', 'phoneme', 'electricity',
                        'scene', 'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
                        'wine_quality', 'us_crime', 'ozone_level', 'webpage', 'coil_2000'])
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

data_temp = ""
for line in logs:
    if line.split("\t")[0] in dataset_names:
        data_temp = line.split("\t")[0]
        performance_flag = False

    if line.startswith("opm_auc,"):
        performance_flag = True
        continue

    if performance_flag:
        tempList = line.replace("\n","").split(",")
        dataset_performance[tempList[0]] = float(tempList[-1].split(" ")[0])/100.
        if line.startswith('RareBoost'):
            list_of_performance.append(dataset_performance)
            dataset_performance = dict()
            performance_flag = False
            if data_temp == 'wine_quality':
                break

# ranking = [i["AdaCC1"] for i in list_of_performance]
# indices, L_sorted = zip(*sorted(enumerate(ranking), key=itemgetter(1)))

# ranked_results = []
# ranked_datasets = []

# for i in indices:
#     ranked_results.append(list_of_performance[i])
#     ranked_datasets.append(datasets_list[i])
# print(ranked_datasets)


# plot_data(ranked_datasets, ranked_results, "Images/", "opm_ranked")
print(len(datasets_list))
print(len(list_of_performance))
print(list_of_performance)
plot_data(datasets_list, list_of_performance, "Images/", "opm_non_ranked")



