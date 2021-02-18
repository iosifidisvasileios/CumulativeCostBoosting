import sys
from collections import defaultdict

import matplotlib
import os

matplotlib.use('Agg')

sys.path.insert(0, 'DataPreprocessing')

dataset_names = set()
list_of_stats = []
logs = open("TableLogs.log", "r")


metric_names = ['accuracy', 'gmean', 'f1score', 'recall', 'opm', 'prc', 'tpr', 'tnr', 'balanced_accuracy', 'auc','opm_prc', 'opm_auc']

new_dataset = False
performance_flag = False
list_of_performance = []
mean_performance = dict()
stdev_performance = dict()

for i in metric_names:
    mean_performance[i] = defaultdict(list)
    stdev_performance[i] = defaultdict(list)
cnt = 0

data_temp = ""
for line in logs:

    if line.split(",")[0] in metric_names:
        metric = line.split(",")[0]
        continue

    tempList = line.replace("\n","").split(",")
    for i in range(1,9):
        mean_performance[metric][tempList[0]].append(float(tempList[i].split("+/-")[0]))
        stdev_performance[metric][tempList[0]].append(float(tempList[i].split("+/-")[1]))


list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaN-CC1', 'AdaN-CC2', 'AdaMEC', 'AdaMEC_Cal', 'CGAda', 'CGAda_Cal', 'AdaCost', 'CSB1',
                       'CSB2', 'AdaC1', 'AdaC2', 'AdaC3', 'RareBoost']

metric_names = ['balanced_accuracy', 'gmean', 'tpr', 'tnr','f1score',  'auc', 'opm_auc']

print(len(mean_performance))
print(mean_performance)

print(metric_names)
for method in list_of_methods:
    print(method)
    for weakL in [0, 1,3,7]:
        line = ' & '+ str(25 + weakL*25)+ ' & '
        for metric in metric_names:
            line += str(mean_performance[metric][method][weakL]) + "$\pm$"+ str(stdev_performance[metric][method][weakL]) + " & "
        line = line[:-3]

        if weakL != 7:
            line += " \\\\ \\cline{2-9}"
        else:
            line += " \\\\ \\hline"
        print(line)




print(metric_names)

for metric in ['balanced_accuracy',]:
    for weakL in [0, 1, 3, 7]:

        best_score =0
        scores = []
        best_method = None
        unsorted_methods = list(list_of_methods)
        for method in list_of_methods:
            scores.append(mean_performance[metric][method][weakL])
            if mean_performance[metric][method][weakL] >= best_score:
                best_score = mean_performance[metric][method][weakL]
                best_method = method

        zipped_lists = zip(scores,unsorted_methods)
        sorted_pairs = sorted(zipped_lists, reverse=True)

        # print(sorted_pairs )
        # print("for metric", metric, "and weak learners=",str(25+25*weakL), "best method = ",  sorted_pairs[:2] )
        # print("for metric", metric, "and weak learners=",str(25+25*weakL),"worst method = " ,  sorted_pairs[-1], 'difference=',      100*(sorted_pairs[0][0] - sorted_pairs[-1][0])/ sorted_pairs[-1][0],
        #       '3rd best method = ', sorted_pairs[3],'min difference = ', 100*(sorted_pairs[0][0] - sorted_pairs[3][0]) / sorted_pairs[3][0], "." )
        # if metric == 'tpr':
        idx_adacc1 =[y[1] for y in sorted_pairs].index('AdaCC2')
        idx =[y[1] for y in sorted_pairs].index('AdaN-CC2')
        # idx =[y[1] for y in sorted_pairs].index('AdaMEC_Cal')
        # print(metric, (weakL*25 +25),100*(sorted_pairs[idx_adacc1][0] - sorted_pairs[idx][0])/sorted_pairs[idx][0])
        print(metric, (weakL*25 +25),sorted_pairs)
