# encoding: utf-8

import matplotlib
from cycler import cycler
from imblearn import metrics
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import os

matplotlib.use('Agg')

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import numpy
import matplotlib.pyplot as plt


def calculate_performance(labels, predictions):
    output = dict()
    output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions[0])
    output["gmean"] = metrics.geometric_mean_score(labels, predictions[0])
    output["accuracy"] = accuracy_score(labels, predictions[0])
    output["f1score"] = f1_score(labels, predictions[0])
    output["recall"] = recall_score(labels, predictions[0])
    output["precision"] = precision_score(labels, predictions[0])

    output["auc"] = roc_auc_score(labels, predictions[1][:, 1])
    output["prc"] = average_precision_score(labels, predictions[1][:, 1])

    tn, fp, fn, tp = confusion_matrix(labels, predictions[0]).ravel()
    output["tpr"] = float(tp) / (float(tp) + float(fn))
    output["tnr"] = float(tn) / (float(tn) + float(fp))
    output["opm"] = (output['gmean'] + output['balanced_accuracy'] + output['f1score'] + output['tpr'] + output[
        "tnr"]) / 5.

    output["opm_prc"] = (output['gmean'] + output['prc'] + output['balanced_accuracy'] + output['f1score'] + output[
        'tpr'] + output["tnr"]) / 6.
    output["opm_auc"] = (output['gmean'] + output['auc'] + output['balanced_accuracy'] + output['f1score'] + output[
        'tpr'] + output["tnr"]) / 6.

    return output


def calculate_fscore(labels, predictions):
    output = dict()
    output["f1score"] = f1_score(labels, predictions)

    return output


def plot_resource_stats_time(methods, list_of_dicts_stats, output_dir, baseL):
    print("time," + ",".join([str(p) for p in baseL]))

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 12})
    plt.grid()
    plt.yscale('log')

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
    plt.xticks(numpy.arange(len(baseL)), [str(k) for k in baseL])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato']
    default_cycler = (cycler(color=colors) +
                      cycler(linestyle=['-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10)),
                                        (0, (5, 1)),
                                        '-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10))])

                      + cycler(marker=['*', 'd', 'x', 'v', 'p', 'X', '^', 's', 'p', 'h', '8']))
    plt.rc('axes', prop_cycle=default_cycler)

    plt.grid(True, axis='y')

    plt.ylabel('Seconds')
    plt.xlabel("Weak Learners")

    for i in range(0, len(methods)):
        y_values = []
        for weak_learners in baseL:
            y_values.append(numpy.mean(list_of_dicts_stats[i][weak_learners]['time']))

        plt.plot(numpy.arange(len(baseL)), y_values, label=methods[i], markersize=12.5)

        my_string = methods[i]

        for pp in range(0, len(baseL)):
            my_string += "," + str(float("%0.2f" % (y_values[pp])))

        print(my_string)

    plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.085), ncol=6)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + "time.png", bbox_inches='tight', dpi=200)


def plot_single_dataset(list_of_names, list_of_dicts, output_dir, baseL):
    items_for_figs = ['accuracy', 'gmean', 'f1score', 'recall', 'opm', 'prc', 'tpr', 'tnr', 'balanced_accuracy', 'auc',
                      'opm_prc', 'opm_auc']
    # items_for_figs = ['accuracy', 'gmean', 'f1score', 'recall', 'opm', 'tpr', 'tnr', 'balanced_accuracy', 'auc']

    for item in items_for_figs:
        print(item + "," + ",".join([str(p) for p in baseL]))
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 12})
        plt.grid()
        plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
        plt.xticks(numpy.arange(len(baseL)), [str(k) for k in baseL])
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato']
        default_cycler = (cycler(color=colors) +
                          cycler(linestyle=['-', (0, (1, 1)), '--', '-.',
                                            (0, (5, 10)),
                                            (0, (5, 1)),
                                            '-', (0, (1, 1)), '--', '-.',
                                            (0, (5, 10))])

                          + cycler(marker=['*', 'd', 'x', 'v', 'p', 'X', '^', 's', 'p', 'h', '8']))
        plt.rc('axes', prop_cycle=default_cycler)

        plt.grid(True, axis='y')

        # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']
        plt.ylabel('%')
        plt.xlabel("Weak Learners")
        for i in range(0, len(list_of_names)):
            y_values = []
            std_values = []
            for weak_learners in baseL:
                y_values.append(numpy.mean(list_of_dicts[i][weak_learners][item]))
                std_values.append(numpy.std(list_of_dicts[i][weak_learners][item]))

            plt.plot(numpy.arange(len(baseL)), y_values, label=list_of_names[i], markersize=12.5)

            my_string = list_of_names[i]
            for pp in range(0, len(baseL)):
                my_string += "," + str(float("%0.2f" % (y_values[pp] * 100))) + " (" + str(
                    float("%0.2f" % (std_values[pp] * 100))) + ")"

            print(my_string)
        plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.085), ncol=6)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(output_dir + item + ".png", bbox_inches='tight', dpi=200)
        plt.clf()


def plot_overall_data(method_names, list_of_dicts, output_dir, baseL):
    # metric_names = ['accuracy', 'gmean', 'f1score', 'recall', 'opm', 'tpr', 'tnr', 'balanced_accuracy', 'auc']
    metric_names = ['accuracy', 'gmean', 'f1score', 'recall', 'opm', 'prc', 'tpr', 'tnr', 'balanced_accuracy', 'auc',
                    'opm_prc', 'opm_auc']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for metric_index in metric_names:
        print(metric_index + "," + ",".join([str(p) for p in baseL]))

        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 12})
        plt.grid()
        plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
        plt.xticks(numpy.arange(len(baseL)), [str(k) for k in baseL])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black', 'orange', 'cyan', 'brown']
        default_cycler = (cycler(color=colors) +
                          cycler(linestyle=['-', (0, (1, 1)), '--', '-.',
                                            (0, (5, 10)),
                                            (0, (5, 1)),
                                            '-', (0, (1, 1)), '--', '-.',
                                            (0, (5, 10)), (0, (5, 1)),
                                            '-', (0, (1, 1))])

                          + cycler(marker=['*', 'd', 'x', 'v', 'p', 'X', '^', 's', 'p', 'h', '8', 'P', '<', ">"]))
        plt.rc('axes', prop_cycle=default_cycler)

        plt.grid(True, axis='y')

        # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']
        plt.ylabel('%')
        plt.xlabel("Weak Learners")

        for method_index in range(0, len(method_names)):
            y_values = []
            std_values = []
            for dataset_index in range(0, len(list_of_dicts)):
                per_dataset_avg = []
                per_dataset_std = []
                for weak_learners in baseL:
                    per_dataset_avg.append(
                        numpy.mean(list_of_dicts[dataset_index][method_index][weak_learners][metric_index]))
                    per_dataset_std.append(
                        numpy.std(list_of_dicts[dataset_index][method_index][weak_learners][metric_index]))
                y_values.append(per_dataset_avg)
                std_values.append(per_dataset_std)

            y_values = [sum(x) / len(x) for x in zip(*y_values)]
            std_values = [sum(x) / len(x) for x in zip(*std_values)]

            plt.plot(numpy.arange(len(baseL)), y_values, label=method_names[method_index], markersize=12.5)
            my_string = method_names[method_index]

            for pp in range(0, len(baseL)):
                my_string += "," + str(float("%0.2f" % (y_values[pp] * 100))) + " (" + str(
                    float("%0.2f" % (std_values[pp] * 100))) + ")"
            print(my_string)

        plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.185), ncol=5)
        plt.savefig(output_dir + metric_index + ".png", bbox_inches='tight', dpi=200)
        plt.clf()


def plot_overall_resource_stats_time(methods, list_of_dicts_stats, output_dir, baseL):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("time," + ",".join([str(p) for p in baseL]))

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 12})
    plt.grid()
    plt.yscale('log')
    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
    plt.xticks(numpy.arange(len(baseL)), [str(k) for k in baseL])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black', 'orange', 'cyan', 'brown']
    default_cycler = (cycler(color=colors) +
                      cycler(linestyle=['-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10)),
                                        (0, (5, 1)),
                                        '-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10)), (0, (5, 1)),
                                        '-', (0, (1, 1))])

                      + cycler(marker=['*', 'd', 'x', 'v', 'p', 'X', '^', 's', 'p', 'h', '8', 'P', '<', ">"]))
    plt.rc('axes', prop_cycle=default_cycler)

    plt.grid(True, axis='y')

    plt.ylabel('Seconds')
    plt.xlabel("Weak Learners")

    for method_index in range(0, len(methods)):
        y_values = []
        for dataset_index in range(0, len(list_of_dicts_stats)):
            per_dataset_avg = []
            for weak_learners in baseL:
                per_dataset_avg.append(
                    numpy.mean(list_of_dicts_stats[dataset_index][method_index][weak_learners]['time']))

            y_values.append(per_dataset_avg)

        y_values = [sum(x) / len(x) for x in zip(*y_values)]

        plt.plot(numpy.arange(len(baseL)), y_values, label=methods[method_index], markersize=12.5)

        my_string = methods[method_index]
        for pp in range(0, len(baseL)):
            my_string += "," + str(float("%0.2f" % (y_values[pp])))

        print(my_string)

    plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.085), ncol=5)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + "time.png", bbox_inches='tight', dpi=200)


def plot_resource_stats_scores(methods, list_of_dicts_stats, output_dir, baseL):
    for item in ['score', 'ratio']:
        print(item + "," + ",".join([str(p) for p in baseL]))

        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 12})
        plt.grid()
        plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
        plt.xticks(numpy.arange(len(baseL)), [str(k) for k in baseL])

        plt.grid(True, axis='y')
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato']
        default_cycler = (cycler(color=colors) +
                          cycler(linestyle=['-', (0, (1, 1)), '--', '-.',
                                            (0, (5, 10)),
                                            (0, (5, 1)),
                                            '-', (0, (1, 1)), '--', '-.',
                                            (0, (5, 10))])

                          + cycler(marker=['*', 'd', 'x', 'v', 'p', 'X', '^', 's', 'p', 'h', '8']))
        plt.rc('axes', prop_cycle=default_cycler)

        plt.ylabel('%')
        plt.xlabel("Weak Learners")

        for i in range(0, len(methods)):
            if methods[i] in ['AdaBoost', 'RareBoost', 'AdaCC1', 'AdaCC2', 'AdaN-CC1', 'AdaN-CC2']:
                continue
            y_values = []
            for weak_learners in baseL:
                if item == 'ratio':
                    y_values.append(numpy.mean(list_of_dicts_stats[i][weak_learners][item]) / 10)
                else:
                    y_values.append(numpy.mean(list_of_dicts_stats[i][weak_learners][item]))

            plt.plot(numpy.arange(len(baseL)), y_values, label=methods[i], markersize=12.5)

            my_string = methods[i]

            for pp in range(0, len(baseL)):
                my_string += "," + str(float("%0.2f" % (y_values[pp])))

            print(my_string)

        plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.085), ncol=6)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(output_dir + item + ".png", bbox_inches='tight', dpi=200)


def plot_overall_resource_stats_scores(methods, list_of_dicts_stats, output_dir, baseL):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for item in ['score', 'ratio']:
        print(item + "," + ",".join([str(p) for p in baseL]))

        list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2',
                           'AdaC3',
                           'RareBoost']

        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 12})
        plt.grid()
        plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
        plt.xticks(numpy.arange(len(baseL)), [str(k) for k in baseL])
        colors = ['c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato']
        default_cycler = (cycler(color=colors) +
                          cycler(linestyle=['-.',
                                            (0, (5, 10)),
                                            (0, (5, 1)),
                                            '-', (0, (1, 1)), '--', '-.',
                                            (0, (5, 10))])

                          + cycler(marker=['v', 'p', 'X', '^', 's', 'p', 'h', '8']))
        plt.rc('axes', prop_cycle=default_cycler)

        plt.grid(True, axis='y')

        plt.ylabel('%')
        plt.xlabel("Weak Learners")

        for method_index in range(0, len(methods)):
            if methods[method_index] in ['AdaBoost', 'RareBoost', 'AdaCC1', 'AdaCC2']:
                continue
            y_values = []
            for dataset_index in range(0, len(list_of_dicts_stats)):
                per_dataset_avg = []
                for weak_learners in baseL:
                    if item == 'ratio':
                        per_dataset_avg.append(
                            numpy.mean(list_of_dicts_stats[dataset_index][method_index][weak_learners][item]) / 10)
                    else:
                        per_dataset_avg.append(
                            numpy.mean(list_of_dicts_stats[dataset_index][method_index][weak_learners][item]))

                y_values.append(per_dataset_avg)

            y_values = [sum(x) / len(x) for x in zip(*y_values)]

            plt.plot(numpy.arange(len(baseL)), y_values, label=methods[method_index], markersize=12.5)

            my_string = methods[method_index]
            for pp in range(0, len(baseL)):
                my_string += "," + str(float("%0.2f" % (y_values[pp])))

            print(my_string)

        plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.085), ncol=6)

        plt.savefig(output_dir + item + ".png", bbox_inches='tight', dpi=200)


def plot_amort_vs_non_amort(methods, results, baseL, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    colors = ['g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato']
    default_cycler = (cycler(color=colors) +
                      cycler(linestyle=[(0, (1, 1)), '--', '-.',
                                        '-',
                                        (0, (5, 1)),
                                        '-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10))]))
    for i in ['TPR_per_round', 'TNR_per_round', 'C_positive_per_round', 'C_negative_per_round', 'alpha',
              'balanced_error']:
        plt.figure(figsize=(7, 7))
        plt.rc('axes', prop_cycle=default_cycler)

        plt.grid(True)
        plt.rcParams.update({'font.size': 10.5})
        for k in range(0, len(methods)):
            res = results[k][i]
            steps = numpy.arange(0, len(res), step=1)
            plt.plot(steps, res, label=methods[k])

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.065), ncol=4, shadow=False, fancybox=True, framealpha=1.0)
        plt.xlabel('Round')
        plt.savefig(directory + i + "_" + str(baseL) + ".png", bbox_inches='tight', dpi=200, shadow=False,
                    fancybox=True, framealpha=.30)


def plot_costs_per_round(methods, results, baseL, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']
    for i in ['neg_class_weights', 'pos_class_weights', 'bal_err', 'alpha']:
        plt.figure(figsize=(7, 7))
        plt.grid(True)
        plt.rcParams.update({'font.size': 10.5})
        for k in range(0, len(methods)):

            if methods[k] == 'RareBoost' and i == 'alpha':
                res_pos = results[k]['alpha_positive']
                res_neg = results[k]['alpha_negative']
                steps = numpy.arange(0, len(res_pos), step=1)
                plt.plot(steps, res_pos, '-', label=methods[k] + "-Pos.", linewidth=1, color=colors[k])
                plt.plot(steps, res_neg, '-', label=methods[k] + "-Neg.", linewidth=1, color=colors[k + 1])


            else:
                res = results[k][i]
                steps = numpy.arange(0, len(res), step=1)
                plt.plot(steps, res, '-', label=methods[k], linewidth=1, color=colors[k])

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.145), ncol=4, shadow=False, fancybox=True, framealpha=1.0)
        plt.xlabel('Round')
        plt.savefig(directory + i + "_" + str(baseL) + ".png", bbox_inches='tight', dpi=200, shadow=False,
                    fancybox=True, framealpha=.30)


def plot_costs_per_round_all_datasets(methods, results, directory, baseL):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              'black']

    for i in ['bal_err', 'alpha', 'pos_class_weights']:
        # for i in ['neg_class_weights', 'pos_class_weights', 'bal_err', 'alpha']:
        print (i)
        steps = numpy.arange(0, baseL, step=1)
        plt.figure(figsize=(7, 7))
        plt.grid(True)
        plt.rcParams.update({'font.size': 10.5})
        default_cycler = (cycler(color=colors) +
                          cycler(linestyle=['-', (0, (1, 1)), '--',
                                            (0, (5, 1)),
                                            '-', (0, (1, 1)), '--', '-.',
                                            (0, (5, 10)), (0, (1, 1))]))

        plt.rc('axes', prop_cycle=default_cycler)

        for jj in range(0, len(methods)):
            if methods[jj] == 'RareBoost' and i == 'alpha':
                res_pos = numpy.array([0. for j in range(0, baseL)])
                res_neg = numpy.array([0. for j in range(0, baseL)])

                for k in results:
                    res_pos += numpy.array(k[jj]['alpha_positive']) / float(len(results))
                    res_neg += numpy.array(k[jj]['alpha_negative']) / float(len(results))

                steps = numpy.arange(0, len(res_pos), step=1)
                plt.plot(steps, res_pos, label=methods[jj] + "-Pos.")
                plt.plot(steps, res_neg, label=methods[jj] + "-Neg.")
                print("RareBoost-Pos.", res_pos )
                print("RareBoost-Neg.", res_neg )

            else:
                res = numpy.array([0. for j in range(0, baseL)])
                for k in results:
                    if res.shape[0] != numpy.array(k[jj][i]).shape[0]:
                        temp_list = list(numpy.array(k[jj][i]))
                        for missing in range(0, res.shape[0] - numpy.array(k[jj][i]).shape[0]):
                            temp_list.append(temp_list[-1])
                        res += numpy.array(temp_list) / float(len(results))
                    else:
                        res += numpy.array(k[jj][i]) / float(len(results))
                # plt.plot(steps, res, '-', label=methods[jj], linewidth=1, color=colors[jj])
                plt.plot(steps, res, label=methods[jj])
                print(methods[jj], res)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.145), ncol=4, shadow=False, fancybox=True, framealpha=1.0)
        plt.xlabel('Round')
        plt.savefig(directory + i + "_" + str(baseL) + ".png", bbox_inches='tight', dpi=200, shadow=False,
                    fancybox=True, framealpha=.30)


def plot_costs_per_round_all_datasets_amort_vs_non_amort(methods, results, directory, baseL):
    if not os.path.exists(directory):
        os.makedirs(directory)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black']
    default_cycler = (cycler(color=colors) +
                      cycler(linestyle=[(0, (1, 1)), '--', '-.',
                                        '-',
                                        (0, (5, 1)),
                                        '-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10))]))

    for i in ['TPR_per_round', 'TNR_per_round', 'C_positive_per_round', 'C_negative_per_round', 'alpha',
              'balanced_error']:
        steps = numpy.arange(0, baseL, step=1)
        plt.figure(figsize=(7, 7))
        plt.rc('axes', prop_cycle=default_cycler)

        plt.grid(True)
        plt.rcParams.update({'font.size': 10.5})

        for jj in range(0, len(methods)):
            res = numpy.array([0. for j in range(0, baseL)])
            for k in results:
                if res.shape[0] != numpy.array(k[jj][i]).shape[0]:
                    temp_list = list(numpy.array(k[jj][i]))
                    for missing in range(0, res.shape[0] - numpy.array(k[jj][i]).shape[0]):
                        temp_list.append(temp_list[-1])
                    res += numpy.array(temp_list) / float(len(results))
                else:
                    res += numpy.array(k[jj][i]) / float(len(results))
            plt.plot(steps, res, label=methods[jj])

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.065), ncol=4, shadow=False, fancybox=True, framealpha=1.0)
        plt.xlabel('Round')
        plt.savefig(directory + i + "_" + str(baseL) + ".png", bbox_inches='tight', dpi=200, shadow=False,
                    fancybox=True, framealpha=.30)


# from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def retrieve_n_class_color_cubic(N):
    '''
    retrive color code for N given classes
    Input: class number
    Output: list of RGB color code
    '''

    # manualy encode the top 8 colors
    # the order is intuitive to be used
    color_list = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (0, 0, 0),
        (1, 1, 1)
    ]

    # if N is larger than 8 iteratively generate more random colors
    np.random.seed(1)  # pre-define the seed for consistency

    interval = 0.5
    while len(color_list) < N:
        the_list = []
        iterator = np.arange(0, 1.0001, interval)
        for i in iterator:
            for j in iterator:
                for k in iterator:
                    if (i, j, k) not in color_list:
                        the_list.append((i, j, k))
        the_list = list(set(the_list))
        np.random.shuffle(the_list)
        color_list.extend(the_list)
        interval = interval / 2.0

    return color_list[:N]


def plot_decision_boundary(model, dim_red_method='pca',
                           X=None, Y=None,
                           xrg=None, yrg=None,
                           Nx=300, Ny=300,
                           scatter_sample=None,
                           figsize=[10, 10], alpha=0.7,
                           random_state=111):
    '''
    Plot decision boundary for any two dimension classification models
        in sklearn.
    Input:
        model: sklearn classification model class - already fitted
                (with "predict" and "predict_proba" method)
        dim_red_method: sklearn dimension reduction model
                (with "fit_transform" and "inverse_transform" method)
        xrg (list/tuple): xrange
        yrg (list/tuple): yrange
        Nx (int): x axis grid size
        Ny (int): y axis grid size
        X (nparray): dataset to project over decision boundary (X)
        Y (nparray): dataset to project over decision boundary (Y)
        figsize, alpha are parameters in matplotlib
    Output:
        matplotlib figure object
    '''

    # check model is legit to use
    try:
        getattr(model, 'predict')
    except:
        print("model do not have method predict 'predict' ")
        return None

    use_prob = False
    try:
        getattr(model, 'predict_proba')
    except:
        print("model do not have method predict 'predict_proba' ")
        use_prob = False

    # convert X into 2D data
    ss, dr_model = None, None
    if X is not None:
        if X.shape[1] == 2:
            X2D = X
        elif X.shape[1] > 2:
            # leverage PCA to dimension reduction to 2D if not already
            ss = StandardScaler()
            if dim_red_method == 'pca':
                dr_model = PCA(n_components=2)
            elif dim_red_method == 'kernal_pca':
                dr_model = KernelPCA(n_components=2,
                                     fit_inverse_transform=True)
            else:
                print('dim_red_method {0} is not supported'.format(
                    dim_red_method))

            X2D = dr_model.fit_transform(ss.fit_transform(X))
        else:
            print('X dimension is strange: {0}'.format(X.shape))
            return None

        # extract two dimension info.
        x1 = X2D[:, 0].min() - 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        x2 = X2D[:, 0].max() + 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        y1 = X2D[:, 1].min() - 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())
        y2 = X2D[:, 1].max() + 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())

    # inti xrg and yrg based on given value
    if xrg is None:
        if X is None:
            xrg = [-10, 10]
        else:
            xrg = [x1, x2]

    if yrg is None:
        if X is None:
            yrg = [-10, 10]
        else:
            yrg = [y1, y2]

    # generate grid, mesh, and X for model prediction
    xgrid = np.arange(xrg[0], xrg[1], 1. * (xrg[1] - xrg[0]) / Nx)
    ygrid = np.arange(yrg[0], yrg[1], 1. * (yrg[1] - yrg[0]) / Ny)

    xx, yy = np.meshgrid(xgrid, ygrid)
    X_full_grid = np.array(list(zip(np.ravel(xx), np.ravel(yy))))

    # initialize figure & axes object
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # get data from model predictions
    if dr_model is None:
        Yp = model.predict(X_full_grid)
        if use_prob:
            Ypp = model.predict_proba(X_full_grid)
        else:
            Ypp = pd.get_dummies(Yp).values
    else:
        X_full_grid_inverse = ss.inverse_transform(
            dr_model.inverse_transform(X_full_grid))

        Yp = model.predict(X_full_grid_inverse)
        if use_prob:
            Ypp = model.predict_proba(X_full_grid_inverse)
        else:
            Ypp = pd.get_dummies(Yp).values

    # retrieve n class from util function
    nclass = Ypp.shape[1]
    colors = np.array(retrieve_n_class_color_cubic(N=nclass))

    # get decision boundary line
    Yp = Yp.reshape(xx.shape)
    Yb = np.zeros(xx.shape)

    Yb[:-1, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb[:-1, :])
    Yb[1:, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb[1:, :])
    Yb[:, :-1] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb[:, :-1])
    Yb[:, 1:] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb[:, 1:])

    # plot decision boundary first
    ax.imshow(Yb, origin='lower', interpolation=None, cmap='Greys',
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=1.0)

    # plot probability surface
    zz = np.dot(Ypp, colors[:nclass, :])
    zz_r = zz.reshape(xx.shape[0], xx.shape[1], 3)
    ax.imshow(zz_r, origin='lower', interpolation=None,
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=alpha)

    # add scatter plot for X & Y if given
    if X is not None:
        # down sample point if needed
        if Y is not None:
            if scatter_sample is not None:
                X2DS, _, YS, _ = train_test_split(X2D, Y, stratify=Y,
                                                  train_size=scatter_sample,
                                                  random_state=random_state)
            else:
                X2DS = X2D
                YS = Y
        else:
            if scatter_sample is not None:
                X2DS, _ = train_test_split(X2D, train_size=scatter_sample,
                                           random_state=random_state)
            else:
                X2DS = X2D

        # convert Y into point color
        if Y is not None:
            # presume Y is labeled from 0 to N-1
            cYS = [colors[i] for i in YS]

        if Y is not None:
            ax.scatter(X2DS[:, 0], X2DS[:, 1], c=cYS)
        else:
            ax.scatter(X2DS[:, 0], X2DS[:, 1])

    # add legend on each class
    colors_bar = []
    for v1 in colors[:nclass, :]:
        v1 = list(v1)
        v1.append(alpha)
        colors_bar.append(v1)

    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors_bar[i],
                              label="Class {k}".format(k=i))
               for i in range(nclass)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),
               loc=2, borderaxespad=0., framealpha=0.5)

    # make the figure nicer
    ax.set_title('Classification decision boundary')
    if dr_model is None:
        ax.set_xlabel('Raw axis X')
        ax.set_ylabel('Raw axis Y')
    else:
        ax.set_xlabel('Dimension reduced axis 1')
        ax.set_ylabel('Dimension reduced axis 2')
    ax.set_xlim(xrg)
    ax.set_ylim(yrg)
    ax.set_xticks(np.arange(xrg[0], xrg[1], (xrg[1] - xrg[0]) / 5.))
    ax.set_yticks(np.arange(yrg[0], yrg[1], (yrg[1] - yrg[0]) / 5.))
    ax.grid(True)

    return fig, ss, dr_model
