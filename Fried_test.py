from collections import defaultdict

import matplotlib
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')
import scipy as sp
import scipy.stats as st


def holm_test(ranks, control=None):
    """
        Performs a Holm post-hoc test using the pivot quantities obtained by a ranking test.
        Tests the hypothesis that the ranking of the control method is different to each of the other methods.

        Parameters
        ----------
        pivots : dictionary_like
            A dictionary with format 'groupname':'pivotal quantity'
        control : string optional
            The name of the control method (one vs all), default None (all vs all)

        Returns
        ----------
        Comparions : array-like
            Strings identifier of each comparison with format 'group_i vs group_j'
        Z-values : array-like
            The computed Z-value statistic for each comparison.
        p-values : array-like
            The associated p-value from the Z-distribution wich depends on the index of the comparison
        Adjusted p-values : array-like
            The associated adjusted p-values wich can be compared with a significance level

        References
        ----------
        O.J. S. Holm, A simple sequentially rejective multiple test procedure, Scandinavian Journal of Statistics 6 (1979) 65–70.
    """
    k = len(ranks)
    values = list(ranks.values())
    keys = list(ranks.keys())

    if not control:
        control_i = values.index(min(values))
    else:
        control_i = keys.index(control)

    comparisons = [keys[control_i] + " vs " + keys[i] for i in range(k) if i != control_i]
    z_values = [abs(values[control_i] - values[i]) for i in range(k) if i != control_i]
    p_values = [2 * (1 - st.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0])))
    adj_p_values = [min(max((k - (j + 1)) * p_values[j] for j in range(i + 1)), 1) for i in range(k - 1)]

    return comparisons, z_values, p_values, adj_p_values

def friedman_test(*args):
    """
        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        -------
        F-value : float
            The computed F-value of the test.
        p-value : float
            The associated p-value from the F-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.

        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """
    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2. for v in row])

    rankings_avg = [sp.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / sp.sqrt(k * (k + 1) / (6. * n)) for r in rankings_avg]

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
                (sp.sum(r ** 2 for r in rankings_avg)) - ((k * (k + 1) ** 2) / float(4)))
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - st.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp

# list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaN-CC1', 'AdaN-CC2', 'AdaMEC', 'AdaMEC_Cal', 'CGAda', 'CGAda_Cal','AdaCost', 'CSB1','CSB2', 'AdaC1', 'AdaC2', 'AdaC3', 'RareBoost']
list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaMEC', 'AdaMEC_Cal', 'CGAda', 'CGAda_Cal','AdaCost', 'CSB1','CSB2', 'AdaC1', 'AdaC2', 'AdaC3', 'RareBoost']

metric_names = ['balanced_accuracy', 'gmean', 'tpr', 'tnr', 'f1score', 'auc', 'opm_auc']
metric_names = [ 'balanced_accuracy']

datasets_list = sorted(['adult', 'wilt', 'credit', 'spam', 'bank', 'musk2', 'isolet',
                        'abalone', 'car_eval_34', 'letter_img', 'protein_homo', 'skin', 'eeg_eye', 'phoneme',
                        'electricity',
                        'scene', 'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid',
                        'thyroid_sick',
                        'wine_quality', 'us_crime', 'ozone_level', 'webpage', 'coil_2000'])

def log_data_for_sig_test(datasets_list,metric_names,list_of_methods, baseL):
    datasets_dict = dict()
    for metric in metric_names:
        allow = True
        dataset_performance = defaultdict(list)
        logs = open("Evaluation.log", "r")

        performance_flag = False
        list_of_performance = []
        cnt = 0
        for line in logs:

            if line.split("\t")[0] in datasets_list:
                allow = True
                continue

            if line.startswith(metric) and allow:
                performance_flag = True
                continue

            if performance_flag and allow:
                tempList = line.replace("\n", "").split(",")
                if tempList[0] in list_of_methods:
                    if metric != 'time':
                        # print(tempList[baseL_index])
                        dataset_performance[tempList[0]].append(float(tempList[baseL].split(" ")[0]) / 100.)
                        # dataset_performance_stdev[tempList[0]].append(float(tempList[baseL_index].split(" ")[1].replace("(","").replace(")",""))/100.)

                    if line.startswith('RareBoost'):
                        # list_of_performance.append(dataset_performance)
                        performance_flag = False
                        # break

        output = open("fried/" + metric + "_fried_data.csv", "w")
        output.write(",".join(list_of_methods) + "\n")
        for idx, dataset in enumerate(datasets_list):
            line_buff = dataset
            for method in list_of_methods:
                line_buff += "," + str(dataset_performance[method][idx])

            output.write(line_buff + "\n")
        output.close()
        logs.close()

for baseL in [1,2,4,8]:
    print(100*"--")
    print(25*baseL)
    log_data_for_sig_test(datasets_list,metric_names,list_of_methods, baseL)
    df = pd.read_csv("fried/balanced_accuracy_fried_data.csv", index_col=0)
    data = np.asarray(df)

    # df_temp = df.transpose()
    # buffer = []
    # for dataset in datasets_list:
    #     buffer.append(list(df_temp[dataset].rank(ascending=False)))
    # print(" & ".join(list_of_methods))
    #
    # for idx, dataset in enumerate(datasets_list):
    #     print(dataset + " & " + " & ".join(map(str, buffer[idx])) + " \\\\")
    #
    # avgs = []
    # for i in range(0, len(list_of_methods)):
    #     avgs.append(np.mean(np.asarray(buffer)[:,i]))
    # print ("avg. & "+" & ".join(format(x, ".2f") for x in avgs) +"\\\\" )

    num_datasets, num_methods = data.shape
    alpha = 0.05
    statistic, p_value, ranking, rank_cmp  = friedman_test(*np.transpose(data))
    ranks = {key: rank_cmp[i] for i, key in enumerate(list(df.columns))}
    # for method, rank in ranks.items():
    #     print(method + ":", "%.2f" % rank)

    comparisons, z_values, p_values, adj_p_values = holm_test(ranks, control="AdaCC1")
    adj_p_values = np.asarray(adj_p_values)
    holm_scores = pd.DataFrame({"p": adj_p_values, "sig": adj_p_values < alpha}, index=comparisons)
    holm_scores = holm_scores['p'].map('{:0.3e}'.format)

    print(holm_scores)


    comparisons, z_values, p_values, adj_p_values = holm_test(ranks, control="AdaCC2")
    adj_p_values = np.asarray(adj_p_values)
    holm_scores = pd.DataFrame({"p": adj_p_values, "sig": adj_p_values < alpha}, index=comparisons)
    holm_scores = holm_scores['p'].map('{:0.3e}'.format)

    print(holm_scores)