import warnings

from DataPreprocessing.load_rain_aus import load_rain_aus

warnings.filterwarnings("ignore")

from time import process_time

from Competitors.AdaMECal import AdaMEC
from DataPreprocessing.load_diabetes import load_diabetes
from DataPreprocessing.load_electricity import load_electricity
from DataPreprocessing.load_phoneme import load_phoneme
from DataPreprocessing.load_speed_dating import load_speed_dating

from Competitors.RareBoost import RareBoost
from DataPreprocessing.load_mat_data import load_mat_data

import pickle
from collections import defaultdict, Counter
import os, sys
import operator
from multiprocessing import Process
from imblearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import time
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
from plot_functions import calculate_performance, plot_single_dataset, plot_overall_data, plot_resource_stats_time, \
    plot_resource_stats_scores, plot_overall_resource_stats_time, plot_overall_resource_stats_scores

sys.path.insert(0, 'DataPreprocessing')


def update_performance_stats(new_stats, output, iterations):
    for i in new_stats:
        output[iterations][i].append(new_stats[i])
    return output


def update_resource_stats(new_stats, output, iterations, method):
    output[iterations]['time'].append(new_stats[0])
    if method not in ['AdaBoost', 'RareBoost', 'AdaCC1', 'AdaCC2', 'AdaN-CC1', 'AdaN-CC2']:
        output[iterations]['score'].append(new_stats[1])
        output[iterations]['ratio'].append(new_stats[2])

    return output


def print_stats(names, stats):
    for i in range(0, len(names)):
        print(names[i] + " " + str(stats[i]))


def run_eval(dataset, folds, iterations, baseL, methods):
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
    elif dataset == "rain_aus":
        X, y, cl_names = load_rain_aus()
        print(y)
    elif dataset == "waveformM":
        X, y, cl_names = load_mat_data(dataset)
    else:
        data = datasets.fetch_datasets()[dataset]
        cl_names = ["feature_" + str(i) for i in range(0, data['data'].shape[1])]
        X = data['data']
        y = data['target']
    y[y != 1] = 0

    unique_attr = set([i.split("?")[0] for i in cl_names])
    print(dataset + "\t" + str(len(unique_attr)) + "\t" + str(f'{sum(abs(y[y == 1])):,}') + "\t" + str(
        f'{len(abs(y[y != 1])):,}') + "\t1:" + str(format(len(abs(y[y != 1])) / sum(y[y == 1]), '.2f')))

    list_of_dicts = []
    list_of_dicts_stats = []

    for t_dict in range(0, len(methods)):
        list_of_dicts.append(defaultdict(dict))
        list_of_dicts_stats.append(defaultdict(dict))

    for weak_learners in baseL:
        for item in list_of_dicts:
            item[weak_learners] = defaultdict(list)

    for weak_learners in baseL:
        for item in list_of_dicts_stats:
            item[weak_learners] = defaultdict(list)
    cnt = 0

    for samples in range(0, iterations):

        sss = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(time.time()))
        for weak_learners in baseL:
            print("iteration=", samples, " weak learners=", weak_learners)

            # for weak_learners in baseL:
            for train_index, test_index in sss.split(X, y):
                cnt += 1

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                processes = []
                for method in methods:
                    p = Process(target=train_and_predict,
                                args=(X_train, y_train, X_test, weak_learners, method, cl_names))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                for index, method in enumerate(methods):
                    with open('temp_preds/' + method, 'rb') as filehandle:
                        list_of_dicts[index] = update_performance_stats(
                            calculate_performance(y_test, pickle.load(filehandle)),
                            list_of_dicts[index],
                            weak_learners
                        )

                    with open('temp_preds/stats_' + method, 'rb') as filehandle:
                        list_of_dicts_stats[index] = update_resource_stats(pickle.load(filehandle),
                                                                           list_of_dicts_stats[index],
                                                                           weak_learners,
                                                                           method
                                                                           )
    plot_single_dataset(methods, list_of_dicts, "Images/Performance/" + dataset + "/", baseL)
    plot_resource_stats_time(methods, list_of_dicts_stats, "Images/Performance/" + dataset + "/Resource/", baseL)
    plot_resource_stats_scores(methods, list_of_dicts_stats, "Images/Performance/" + dataset + "/Resource/", baseL)
    return list_of_dicts, list_of_dicts_stats

def train_and_predict(X_train, y_train, X_test, base_learners, method, cl_names):
    if method == 'AdaBoost':
        t1_start = process_time()
        clf = AdaCost(n_estimators=base_learners, algorithm=method)
        clf.fit(X_train, y_train)
        t1_stop = process_time()
        oveall_time = t1_stop - t1_start

        with open('temp_preds/stats_' + method, 'wb') as filehandle:
            pickle.dump([oveall_time], filehandle)

        with open('temp_preds/' + method, 'wb') as filehandle:
            pickle.dump([clf.predict(X_test), clf.predict_proba(X_test)], filehandle)
        return

    elif 'AdaCC' in method:
        t1_start = process_time()

        clf = AdaCC(n_estimators=base_learners, algorithm=method)
        clf.fit(X_train, y_train)
        t1_stop = process_time()
        oveall_time = t1_stop - t1_start

        with open('temp_preds/stats_' + method, 'wb') as filehandle:
            pickle.dump([oveall_time], filehandle)

        with open('temp_preds/' + method, 'wb') as filehandle:
            # pickle.dump(clf.predict(X_test), filehandle)
            pickle.dump([clf.predict(X_test), clf.predict_proba(X_test)], filehandle)

        return

    elif 'AdaN-CC' in method:
        t1_start = process_time()

        clf = AdaCC(n_estimators=base_learners, algorithm=method.replace("N-", ""), amortised=False)
        clf.fit(X_train, y_train)

        t1_stop = process_time()
        oveall_time = t1_stop - t1_start

        with open('temp_preds/stats_' + method, 'wb') as filehandle:
            pickle.dump([oveall_time], filehandle)

        with open('temp_preds/' + method, 'wb') as filehandle:
            # pickle.dump(clf.predict(X_test), filehandle)
            pickle.dump([clf.predict(X_test), clf.predict_proba(X_test)], filehandle)
        return

    elif 'AdaMEC' in method:
        counter_dict = Counter(list(y_train))

        majority = max(counter_dict.items(), key=operator.itemgetter(1))[0]
        minority = max(counter_dict.items(), key=operator.itemgetter(0))[0]
        ratios = [1., 2., 3., 4., 5., 6., 7, 8., 9., 10.]

        t1_start = process_time()

        clf = AdaMEC(n_estimators=base_learners, algorithm=method)
        clf.fit(X_train, y_train)
        best_score = -1
        best_idx = 0
        for idx, cost in enumerate(ratios):
            class_weight = {minority: 1, majority: cost / 10.}
            clf.set_costs(y_train, class_weight)
            score = f1_score(y_train, clf.predict(X_train))
            if best_score < score:
                best_idx = idx
                best_score = score
        class_weight = {minority: 1, majority: ratios[best_idx] / 10.}

        clf.set_costs(y_train, class_weight)

        t1_stop = process_time()
        oveall_time = t1_stop - t1_start

        with open('temp_preds/stats_' + method, 'wb') as filehandle:
            pickle.dump([oveall_time, best_score, ratios[best_idx]], filehandle)

        with open('temp_preds/' + method, 'wb') as filehandle:
            # pickle.dump(clf.predict(X_test), filehandle)
            pickle.dump([clf.predict(X_test), clf.predict_proba(X_test)], filehandle)
        return

    elif method == 'RareBoost':
        t1_start = process_time()

        clf = RareBoost(n_estimators=base_learners)
        clf.fit(X_train, y_train)
        t1_stop = process_time()
        oveall_time = t1_stop - t1_start

        with open('temp_preds/stats_' + method, 'wb') as filehandle:
            pickle.dump([oveall_time], filehandle)

        with open('temp_preds/' + method, 'wb') as filehandle:
            # pickle.dump(clf.predict(X_test), filehandle)
            pickle.dump([clf.predict(X_test), clf.predict_proba(X_test)], filehandle)

        return

    else:
        counter_dict = Counter(list(y_train))

        majority = max(counter_dict.items(), key=operator.itemgetter(1))[0]
        minority = max(counter_dict.items(), key=operator.itemgetter(0))[0]
        ratios = [1., 2., 3., 4., 5., 6., 7, 8., 9., 10.]

        processes = []
        for ratio in ratios:
            p = Process(target=train_competitors,
                        args=(X_train, y_train, base_learners, method, majority, minority, ratio))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        best_score = -1
        best_ratio = 0
        oveall_time = 0
        for ratio in ratios:
            if os.path.exists('temp_preds/' + method + str(ratio)):
                with open('temp_preds/' + method + str(ratio), 'rb') as filehandle:
                    temp = pickle.load(filehandle)
                    oveall_time += temp[2]
                    if temp[0] > best_score:
                        best_ratio = ratio
                        best_score = temp[0]
                        clf = temp[1]

            if os.path.exists('temp_preds/' + method + str(ratio)):
                os.remove('temp_preds/' + method + str(ratio))

        with open('temp_preds/stats_' + method, 'wb') as filehandle:
            pickle.dump([oveall_time, best_score, best_ratio], filehandle)

        with open('temp_preds/' + method, 'wb') as filehandle:
            # pickle.dump(clf.predict(X_test), filehandle)
            pickle.dump([clf.predict(X_test), clf.predict_proba(X_test)], filehandle)

        return


def train_competitors(X_train, y_train, base_learners, method, maj, min, ratio):
    try:
        out = []
        t1_start = process_time()
        clf = AdaCost(n_estimators=base_learners, algorithm=method, class_weight={min: 1, maj: ratio / 10.})
        clf.fit(X_train, y_train)
        t1_stop = process_time()

        out.append(f1_score(y_train, clf.predict(X_train)))
        out.append(clf)
        out.append(t1_stop - t1_start)
        with open('temp_preds/' + method + str(ratio), 'wb') as filehandle:
            pickle.dump(out, filehandle)
    except:
        return


if __name__ == '__main__':
    if not os.path.exists("temp_preds"):
        os.makedirs("temp_preds")
    if not os.path.exists("Images"):
        os.makedirs("Images")
    if not os.path.exists("Images/Performance/"):
        os.makedirs("Images/Performance/")

    baseL = [25, 50, 75, 100, 125, 150, 175, 200]

    dicts_for_plots = []
    dicts_for_plots_stats = []

    list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3', 'RareBoost']
    # list_of_methods = [ 'AdaN-CC1', 'AdaN-CC2']

    # datasets_list = sorted(['mushroom', 'adult', 'wilt', 'credit', 'spam', 'bank', 'landsatM', 'musk2', 'isolet',
    #                         'spliceM', 'semeion_orig', 'waveformM', 'abalone', 'car_eval_34', 'letter_img',
    #                         'skin', 'eeg_eye', 'phoneme', 'electricity', 'scene',  # 'kdd' ,'diabetes',
    #                         'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
    #                         'wine_quality', 'us_crime', 'protein_homo', 'ozone_level', 'webpage', 'coil_2000'])

    datasets_list = sorted(['adult', 'wilt', 'credit', 'spam', 'bank', 'musk2', 'isolet',
                            'abalone', 'car_eval_34', 'letter_img', 'protein_homo', 'skin', 'eeg_eye', 'phoneme', 'electricity',
                            'scene', 'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
                            'wine_quality', 'us_crime', 'ozone_level', 'webpage', 'coil_2000'])

    datasets_list = ['rain_aus']
    for dataset in datasets_list:
        if dataset == 'kdd' or dataset == 'skin' or dataset == 'diabetes' or \
                dataset == 'protein_homo' or dataset == 'webpage' or dataset == 'isolet':
            dataset_dict1, dataset_dict2 = run_eval(dataset=dataset, folds=5, iterations=1, baseL=baseL,
                                                    methods=list_of_methods)
        else:
            dataset_dict1, dataset_dict2 = run_eval(dataset=dataset, folds=5, iterations=10, baseL=baseL,
                                                    methods=list_of_methods)

        dicts_for_plots.append(dataset_dict1)
        dicts_for_plots_stats.append(dataset_dict2)

    plot_overall_data(list_of_methods, dicts_for_plots, "Images/Performance/Overall/", baseL)
    plot_overall_resource_stats_time(list_of_methods, dicts_for_plots_stats, "Images/Performance/Overall/Resource/",
                                     baseL)
    plot_overall_resource_stats_scores(list_of_methods, dicts_for_plots_stats, "Images/Performance/Overall/Resource/",
                                       baseL)
