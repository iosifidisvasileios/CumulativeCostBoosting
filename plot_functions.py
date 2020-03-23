# encoding: utf-8

import matplotlib
from imblearn import metrics
from sklearn.metrics import balanced_accuracy_score
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

    tn, fp, fn, tp = confusion_matrix(labels, predictions[0]).ravel()
    output["tpr"] = float(tp) / (float(tp) + float(fn))
    output["tnr"] = float(tn) / (float(tn) + float(fp))
    output["opm"] = (output['gmean'] + output['balanced_accuracy'] + output['f1score'] + output['recall'] + output["accuracy"] ) / 5.

    return output


def calculate_fscore(labels, predictions):
    output = dict()
    output["f1score"] = f1_score(labels, predictions)

    return output


def plot_single_dataset(list_of_names, list_of_dicts, output_dir, baseL):
    # items_for_figs = ['accuracy', 'gmean', 'f1score', 'recall', 'opm', 'tpr', 'tnr','balanced_accuracy', 'auc_score']
    items_for_figs = ['accuracy', 'gmean', 'f1score', 'recall', 'opm', 'tpr', 'tnr', 'balanced_accuracy']

    for item in items_for_figs:
        print(item + "," + ",".join([str(p) for p in baseL]))
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 12})
        plt.grid()
        plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
        plt.xticks(numpy.arange(len(baseL)), [str(k) for k in baseL])

        plt.grid(True, axis='y')

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']
        plt.ylabel('%')
        plt.xlabel("Weak Learners")
        for i in range(0, len(list_of_names)):
            y_values = []
            std_values = []
            for weak_learners in baseL:
                y_values.append(numpy.mean(list_of_dicts[i][weak_learners][item]))
                std_values.append(numpy.std(list_of_dicts[i][weak_learners][item]))

            plt.plot(numpy.arange(len(baseL)), y_values, label=list_of_names[i], color=colors[i])

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
    metric_names = ['accuracy', 'gmean', 'f1score', 'recall', 'opm', 'tpr', 'tnr', 'balanced_accuracy']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for metric_index in metric_names:
        print(metric_index + "," + ",".join([str(p) for p in baseL]))

        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 12})
        plt.grid()
        plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
        plt.xticks(numpy.arange(len(baseL)), [str(k) for k in baseL])

        plt.grid(True, axis='y')

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']
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

            plt.plot(numpy.arange(len(baseL)), y_values, label=method_names[method_index], color=colors[method_index])
            my_string = method_names[method_index]

            for pp in range(0, len(baseL)):
                my_string += "," + str(float("%0.2f" % (y_values[pp] * 100))) + " (" + str(
                    float("%0.2f" % (std_values[pp] * 100))) + ")"
            print(my_string)

        plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.085), ncol=6)
        plt.savefig(output_dir + metric_index + ".png", bbox_inches='tight', dpi=200)
        plt.clf()

def plot_costs_per_round(methods, results, baseL, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']
    for i in ['pos_class_weights', 'neg_class_weights', 'bal_err', 'alpha']:
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

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato', 'indigo', 'lightskyblue']

    for i in ['pos_class_weights', 'neg_class_weights', 'bal_err', 'alpha']:
        steps = numpy.arange(0, baseL, step=1)
        plt.figure(figsize=(7, 7))
        plt.grid(True)
        plt.rcParams.update({'font.size': 10.5})

        for jj in range(0, len(methods)):
            if methods[jj] == 'RareBoost' and i == 'alpha':
                res_pos = numpy.array([0. for j in range(0, baseL)])
                res_neg = numpy.array([0. for j in range(0, baseL)])

                for k in results:
                    res_pos += numpy.array(k[jj]['alpha_positive']) / float(len(results))
                    res_neg += numpy.array(k[jj]['alpha_negative']) / float(len(results))

                steps = numpy.arange(0, len(res_pos), step=1)
                plt.plot(steps, res_pos, '-', label=methods[jj] + "-Pos.", linewidth=1, color=colors[jj])
                plt.plot(steps, res_neg, '-', label=methods[jj] + "-Neg.", linewidth=1, color=colors[jj + 1])

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
                plt.plot(steps, res, '-', label=methods[jj], linewidth=1, color=colors[jj])

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.145), ncol=4, shadow=False, fancybox=True, framealpha=1.0)
        plt.xlabel('Round')
        plt.savefig(directory + i + "_" + str(baseL) + ".png", bbox_inches='tight', dpi=200, shadow=False,
                    fancybox=True, framealpha=.30)
