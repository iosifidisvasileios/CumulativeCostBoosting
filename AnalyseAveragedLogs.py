from collections import defaultdict
from statistics import mean
from cycler import cycler

import matplotlib
import os
import matplotlib.pyplot as plt
import numpy

matplotlib.use('Agg')

def plot_the_lists(mean, stDev, output_dir):
    baseL = [25, 50, 75, 100, 125, 150, 175, 200]
    list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2',
                       'AdaC3', 'RareBoost']
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 12})

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray', 'peru', 'hotpink', 'tomato']
    default_cycler = (cycler(color=colors) +
                      cycler(linestyle=['-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10)),
                                        (0, (5, 1)),
                             '-', (0, (1, 1)), '--', '-.',
                                        (0, (5, 10))])

                      +cycler(marker=['*', 'd', 'x', 'v', 'p', 'X', '^', 's', 'p', 'h', '8']))
    plt.rc('axes', prop_cycle=default_cycler)

    plt.grid()
    # plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
    plt.xticks(numpy.arange(len(baseL)), [str(k) for k in baseL])
    plt.grid(True, axis='y')

    # plt.ylabel('%')
    plt.xlabel("Weak Learners")

    for idx, method in enumerate(list_of_methods):
        plt.errorbar(numpy.arange(len(baseL)), mean[method],   markersize=12.5, yerr=stDev[method], label=method,  )

    plt.legend(loc='upper center', bbox_to_anchor=(0.49, 1.08), ncol=6)

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    plt.savefig(output_dir + ".png", bbox_inches='tight', dpi=200)
    plt.clf()

list_of_methods = ['AdaBoost', 'AdaCC1', 'AdaCC2', 'AdaMEC', 'AdaCost', 'CSB1', 'CSB2', 'AdaC1', 'AdaC2', 'AdaC3','RareBoost']
measures = ['gmean', 'f1score', 'tpr', 'tnr','balanced_accuracy']

datasets_list = sorted([ 'adult', 'wilt', 'credit', 'spam', 'bank', 'musk2', 'isolet',
                         'abalone', 'car_eval_34', 'letter_img',
                        'skin', 'eeg_eye', 'phoneme', 'electricity', 'scene',
                        'mammography', 'optical_digits', 'pen_digits', 'satimage', 'sick_euthyroid', 'thyroid_sick',
                        'wine_quality', 'us_crime',  'ozone_level', 'webpage', 'coil_2000'])

banned_sets = ['mushroom', 'landsatM','spliceM', 'semeion_orig', 'waveformM','protein_homo',]
[0., 0., 0., 0., 0., 0., 0., 0.]
opm = defaultdict(list)
opm_stdev = defaultdict(list)
for method in list_of_methods:
    opm[method] = [0., 0., 0., 0., 0., 0., 0., 0.]
    opm_stdev[method] = [0., 0., 0., 0., 0., 0., 0., 0.]

for item in measures:
    allow = True
    print(item)
    dataset_performance_avg = defaultdict(list)
    dataset_performance_stdev_avg = defaultdict(list)

    for baseL_index in range(1, 9):

        dataset_performance = defaultdict(list)
        dataset_performance_stdev = defaultdict(list)

        logs = open("Evaluation.log", "r")

        performance_flag = False
        list_of_performance = []
        cnt = 0
        for line in logs:

            if line.split("\t")[0] in banned_sets:
                allow = False
                print(line)
                continue

            if line.split("\t")[0] in datasets_list:
                allow = True
                print(baseL_index, line)
                continue

            if line.startswith(item) and allow:
                performance_flag = True
                continue

            if performance_flag and allow:
                tempList = line.replace("\n","").split(",")
                if tempList[0] in list_of_methods:
                    dataset_performance[tempList[0]].append(float(tempList[baseL_index].split(" ")[0])/100.)
                    dataset_performance_stdev[tempList[0]].append(float(tempList[baseL_index].split(" ")[1].replace("(","").replace(")",""))/100.)
                    if line.startswith('RareBoost'):
                        # list_of_performance.append(dataset_performance)
                        performance_flag = False
                        # break
        for method in list_of_methods:
            dataset_performance_avg[method].append(mean(dataset_performance[method]))
            dataset_performance_stdev_avg[method].append(mean(dataset_performance_stdev[method]))

    for method in list_of_methods:
        opm[method] = [dataset_performance_avg[method][j]/5. + opm[method][j] for j in range(0, 8)]
        opm_stdev[method] = [dataset_performance_stdev_avg[method][j]/5. + opm_stdev[method][j] for j in range(0, 8)]

    print(dataset_performance_avg)
    print(len(dataset_performance_avg['AdaBoost']))
    plot_the_lists(dataset_performance_avg, dataset_performance_stdev_avg,"Images/" + item)

    logs.close()

plot_the_lists(opm, opm_stdev,"Images/opm" )


