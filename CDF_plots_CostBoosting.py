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

from imblearn import datasets
from multiprocessing import Process
from sklearn.metrics import f1_score, balanced_accuracy_score

from Competitors.RareBoost import RareBoost
from Competitors.AdaC1C3 import AdaCost
from DataPreprocessing.load_mat_data import load_mat_data
from AdaCC import AdaCC

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, 'DataPreprocessing')

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
        data = datasets.fetch_datasets()[dataset]
        cl_names = ["feature_" + str(i) for i in range(0, data['data'].shape[1])]
        X = data['data']
        y = data['target']

    y[y != 1] = 0

    list_of_scores = []
    processes = []

    for method in methods:
        p = Process(target=train_classifier, args=(X, y, base_learners, method, cl_names))  # Passing the list
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for method in methods:
        with open('temp/' + method, 'rb') as filehandle:
            # read the data as binary data stream
            list_of_scores.append(pickle.load(filehandle))

    y[y != 1] = -1

    for idx in range(0, len(list_of_scores)):
        list_of_scores[idx] = numpy.array(list_of_scores[idx]) * y

    overall_confs = []
    positive_confs = []
    negative_confs = []

    for conf in list_of_scores:
        overall_confs.append(conf)
        positive_confs.append(conf[y == 1])
        negative_confs.append(conf[y == -1])

    num_bins = 40
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    plt.rcParams.update({'font.size': 12})
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato']
    default_cycler = (cycler(color=colors) +
                      cycler(linestyle=['-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10)),
                                        (0, (5, 1)),
                             '-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10))]))

    ax1.set_prop_cycle(default_cycler)
    ax2.set_prop_cycle(default_cycler)
    ax3.set_prop_cycle(default_cycler)

    ax1.set_title("Positive CDF")
    ax1.grid(True)
    ax1.set_xlim(-1, 1)
    ax2.set_xlim(-1, 1)
    ax3.set_xlim(-1, 1)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']

    output = defaultdict(list)

    for idx in range(0, len(positive_confs)):
        pos_conf = positive_confs[idx]
        counts_positives, bin_edges_positives = numpy.histogram(pos_conf, bins=num_bins, normed=True)
        cdf_positives = numpy.cumsum(counts_positives)
        # ax1.plot(bin_edges_positives[:-1], cdf_positives / cdf_positives[-1], label=methods[idx],color=colors[idx])
        ax1.plot(bin_edges_positives[:-1], cdf_positives / cdf_positives[-1], label=methods[idx])
        output[methods[idx]].append(bin_edges_positives[:-1])
        output[methods[idx]].append(cdf_positives)

    # ax1.legend(loc='best')
    ax1.set_xlabel("Margin")

    ax1.set_ylabel("Cumulative Distribution")
    ax1.axhline(0, color='black')
    ax1.axvline(0, color='black')

    ax2.grid(True)

    ax2.axhline(0, color='black')
    ax2.axvline(0, color='black')
    ax2.set_title("Negative CDF")

    for idx in range(0, len(negative_confs)):
        if idx == 0:
            ax2.set_ylabel("Cumulative Distribution")
            ax2.set_xlabel("Margin")

        neg_conf = negative_confs[idx]
        counts_negatives, bin_edges_negatives = numpy.histogram(neg_conf, bins=num_bins, normed=True)
        cdf_negatives = numpy.cumsum(counts_negatives)
        # ax2.plot(bin_edges_negatives[:-1], cdf_negatives / cdf_negatives[-1], label=methods[idx],color=colors[idx])
        ax2.plot(bin_edges_negatives[:-1], cdf_negatives / cdf_negatives[-1], label=methods[idx])
        output[methods[idx]].append(bin_edges_negatives[:-1])
        output[methods[idx]].append(cdf_negatives)

    ax3.grid(True)

    ax3.axhline(0, color='black')
    ax3.axvline(0, color='black')
    ax3.set_title("Overall CDF")

    for idx in range(0, len(negative_confs)):
        if idx == 0:
            ax3.set_ylabel("Cumulative Distribution")
            ax3.set_xlabel("Margin")

        over_conf = overall_confs[idx]
        counts_overall, bin_edges_overall = numpy.histogram(over_conf, bins=num_bins, normed=True)
        cdf_overall = numpy.cumsum(counts_overall)
        # ax3.plot(bin_edges_overall[:-1], cdf_overall / cdf_overall[-1], label=methods[idx], color=colors[idx])
        ax3.plot(bin_edges_overall[:-1], cdf_overall / cdf_overall[-1], label=methods[idx])
        output[methods[idx]].append(bin_edges_overall[:-1])
        output[methods[idx]].append(cdf_overall)

    plt.legend(loc='upper center', bbox_to_anchor=(-0.7, 1.305), ncol=5)

    if not os.path.exists("Images/cdf_plots/" + dataset):
        os.makedirs("Images/cdf_plots/" + dataset)

    plt.savefig("Images/cdf_plots/" + dataset + "/cdf_" + str(base_learners) + ".png", bbox_inches='tight',
                dpi=200)
    return output


def train_classifier(X_train, y_train, base_learners, method, cl_names):
    if method == 'AdaBoost':
        clf = AdaCost(algorithm='AdaBoost', n_estimators=base_learners)
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
                clf = AdaCost(n_estimators=base_learners, algorithm=method,
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

    with open('temp/' + method, 'wb') as filehandle:
        pickle.dump(numpy.asarray(clf.get_confidence_scores(X_train)), filehandle)


def plot_overall_data(method_names, list_dataset_dicts, base_learners, output_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    plt.rcParams.update({'font.size': 12})

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato']
    default_cycler = (cycler(color=colors) +
                      cycler(linestyle=['-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10)),
                                        (0, (5, 1)),
                             '-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10))]))

    ax1.set_prop_cycle(default_cycler)
    ax2.set_prop_cycle(default_cycler)
    ax3.set_prop_cycle(default_cycler)

    ax1.set_xlim(-1, 1)
    ax2.set_xlim(-1, 1)
    ax3.set_xlim(-1, 1)

    ax1.set_title("Positive CDF")
    ax1.grid(True)
    ax1.set_xlabel("Margin")
    ax1.set_ylabel("Cumulative Distribution")
    ax1.axhline(0, color='black')
    ax1.axvline(0, color='black')
    ax2.grid(True)
    ax2.axhline(0, color='black')
    ax2.axvline(0, color='black')
    ax2.set_title("Negative CDF")
    ax2.set_ylabel("Cumulative Distribution")
    ax2.set_xlabel("Margin")
    ax3.grid(True)
    ax3.axhline(0, color='black')
    ax3.axvline(0, color='black')
    ax3.set_title("Overall CDF")
    ax3.set_ylabel("Cumulative Distribution")
    ax3.set_xlabel("Margin")
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cnt = 0
    for method_idx in method_names:

        avg_edges_overall = []
        avg_cdf_overall = []
        avg_edges_pos = []
        avg_cdf_pos = []
        avg_edges_neg = []
        avg_cdf_neg = []

        for dataset_idx in list_dataset_dicts:
            data_dict = dataset_idx
            avg_edges_pos.append(data_dict[method_idx][0])
            avg_cdf_pos.append(data_dict[method_idx][1])
            avg_edges_neg.append(data_dict[method_idx][2])
            avg_cdf_neg.append(data_dict[method_idx][3])
            avg_edges_overall.append(data_dict[method_idx][4])
            avg_cdf_overall.append(data_dict[method_idx][5])

        avg_edges_pos = [sum(x) / len(x) for x in zip(*avg_edges_pos)]
        avg_cdf_pos = [sum(x) / len(x) for x in zip(*avg_cdf_pos)]
        avg_edges_neg = [sum(x) / len(x) for x in zip(*avg_edges_neg)]
        avg_cdf_neg = [sum(x) / len(x) for x in zip(*avg_cdf_neg)]
        avg_edges_overall = [sum(x) / len(x) for x in zip(*avg_edges_overall)]
        avg_cdf_overall = [sum(x) / len(x) for x in zip(*avg_cdf_overall)]
        #
        # ax1.plot(avg_edges_pos, avg_cdf_pos / avg_cdf_pos[-1], label=method_idx, color=colors[cnt])
        # ax2.plot(avg_edges_neg, avg_cdf_neg / avg_cdf_neg[-1], label=method_idx, color=colors[cnt])
        # ax3.plot(avg_edges_overall, avg_cdf_overall / avg_cdf_overall[-1], label=method_idx, color=colors[cnt])
        ax1.plot(avg_edges_pos, avg_cdf_pos / avg_cdf_pos[-1], label=method_idx )
        ax2.plot(avg_edges_neg, avg_cdf_neg / avg_cdf_neg[-1], label=method_idx )
        ax3.plot(avg_edges_overall, avg_cdf_overall / avg_cdf_overall[-1], label=method_idx )
        cnt += 1

    plt.legend(loc='upper center', bbox_to_anchor=(-0.7, 1.305), ncol=5)
    plt.savefig(output_dir + "cdf_" + str(base_learners) + ".png", bbox_inches='tight', dpi=200)


if __name__ == '__main__':

    if not os.path.exists("Images"):
        os.makedirs("Images")
    if not os.path.exists("Images/cdf_plots/"):
        os.makedirs("Images/cdf_plots/")
    if not os.path.exists("temp"):
        os.makedirs("temp")

    list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3', 'RareBoost']
    #
    # datasets_list = sorted(['mushroom', 'adult', 'wilt', 'credit', 'spam', 'bank', 'landsatM', 'musk2', 'isolet',
    #                         'spliceM', 'semeion_orig', 'waveformM', 'abalone', 'car_eval_34', 'letter_img',
    #                         'skin', 'eeg_eye', 'phoneme', 'electricity', 'scene',  # 'kdd' ,'diabetes',
    #                         'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
    #                         'wine_quality', 'us_crime', 'protein_homo', 'ozone_level', 'webpage', 'coil_2000'])

    datasets_list = sorted(['adult', 'wilt', 'credit', 'spam', 'bank', 'musk2', 'isolet',
                            'abalone', 'car_eval_34', 'letter_img', 'protein_homo', 'skin', 'eeg_eye', 'phoneme', 'electricity',
                            'scene', 'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
                            'wine_quality', 'us_crime', 'ozone_level', 'webpage', 'coil_2000'])
    for baseL in [25, 200]:
        overall_list = []
        for dataset in datasets_list:
            print(dataset, baseL)
            overall_list.append(run_eval(dataset=dataset, base_learners=baseL, methods=list_of_methods))
        plot_overall_data(list_of_methods, overall_list, baseL, "Images/cdf_plots/Overall/")
