import warnings
warnings.filterwarnings("ignore")

import operator

from sklearn.metrics import f1_score

from Competitors.AdaC1C3 import AdaCost
from Competitors.RareBoost import RareBoost
from DataPreprocessing.load_diabetes import load_diabetes
from DataPreprocessing.load_electricity import load_electricity
from DataPreprocessing.load_phoneme import load_phoneme
from DataPreprocessing.load_speed_dating import load_speed_dating


from DataPreprocessing.load_mat_data import load_mat_data

import pickle
from collections import Counter
import os, sys
from multiprocessing import Process
from imblearn import datasets
from AdaCC import AdaCC
from DataPreprocessing.load_adult import load_adult
from DataPreprocessing.load_wilt import load_wilt
from DataPreprocessing.load_mushroom import load_mushroom
from DataPreprocessing.load_eeg_eye import load_eeg_eye
from DataPreprocessing.load_spam import load_spam
from DataPreprocessing.load_skin import load_skin
from DataPreprocessing.load_credit import load_credit
from DataPreprocessing.load_kdd import load_kdd
from DataPreprocessing.load_bank import load_bank
from plot_functions import plot_costs_per_round, \
    plot_costs_per_round_all_datasets, plot_amort_vs_non_amort, plot_costs_per_round_all_datasets_amort_vs_non_amort

sys.path.insert(0, 'DataPreprocessing')


def update_stats(new_stats, RareBoostFlag):
    output = dict()

    output['TPR_per_round'] = new_stats[0]
    output['TNR_per_round'] = new_stats[1]
    output['C_positive_per_round'] = new_stats[2]
    output['C_negative_per_round'] = new_stats[3]

    return output


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
    print("===============-- " + dataset + " --===============")

    processes = []
    for method in methods:
        p = Process(target=train_and_predict, args=(X, y, baseL, method))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    list_of_dicts = []

    for method in methods:
        with open('temp_preds_adaac/' + method, 'rb') as filehandle:
            list_of_dicts.append(update_stats(pickle.load(filehandle), False))

    plot_amort_vs_non_amort(methods, list_of_dicts, baseL, "Images/Amort_vs_non_amort/" + dataset + "/")
    return list_of_dicts


def train_and_predict(X_train, y_train, base_learners, method):
    if 'AdaCC' in method:
        clf = AdaCC(n_estimators=base_learners, algorithm=method, debug=True)
        clf.fit(X_train, y_train)
    elif 'AdaN-CC' in method:
        clf = AdaCC(n_estimators=base_learners, algorithm=method.replace("N-", ""), debug=True, amortised=False)
        clf.fit(X_train, y_train)

    with open('temp_preds_adaac/' + method, 'wb') as filehandle:
        # pickle.dump([clf._class_weights_pos, clf._class_weights_neg, clf.training_error, clf.estimator_alphas_], filehandle)
        pickle.dump([clf._tpr, clf._tnr,  clf.cost_pos, clf.cost_neg], filehandle)


if __name__ == '__main__':

    if not os.path.exists("Images"):
        os.makedirs("Images")

    if not os.path.exists("Images/Amort_vs_non_amort"):
        os.makedirs("Images/Amort_vs_non_amort")

    if not os.path.exists("Amort_vs_non_amort"):
        os.makedirs("Amort_vs_non_amort")

    if not os.path.exists("temp_preds_adaac"):
        os.makedirs("temp_preds_adaac")

    baseLearners = [200]
    # list_of_methods = ['AdaBoost', 'AdaAC1', 'AdaAC2', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3', 'RareBoost']
    list_of_methods = ['AdaCC1', 'AdaCC2', 'AdaN-CC1', 'AdaN-CC2',]

    datasets_list = sorted(['mushroom', 'adult', 'wilt', 'credit', 'spam', 'bank', 'landsatM', 'musk2', 'isolet',
                            'spliceM', 'semeion_orig', 'waveformM', 'abalone', 'car_eval_34', 'letter_img',
                            'skin', 'eeg_eye', 'phoneme', 'electricity', 'scene',  # 'kdd' ,'diabetes',
                            'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
                            'wine_quality', 'us_crime', 'protein_homo', 'ozone_level', 'webpage', 'coil_2000'])

    for baseL in baseLearners:
        dicts_for_plots = []
        for dataset in datasets_list:
            dicts_for_plots.append(run_eval(dataset=dataset, baseL=baseL, methods=list_of_methods))

        plot_costs_per_round_all_datasets_amort_vs_non_amort(list_of_methods, dicts_for_plots, "Images/Amort_vs_non_amort/Overall/", baseL)
