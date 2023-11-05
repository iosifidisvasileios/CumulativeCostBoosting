from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, recall_score, precision_score, \
	roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from imbalanced_ensemble import datasets
from cumulative_cost_boosting import AdaCC

def calculate_performance(labels, predictions):
	output = dict()
	output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions[0])
	output["gmean"] = geometric_mean_score(labels, predictions[0])
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

	return output


datasets_all = ['mammography', 'optical_digits', 'pen_digits', 'satimage', 'thyroid_sick', 'sick_euthyroid']

for dataset in datasets_all:

	data = datasets.fetch_datasets()[dataset]
	cl_names = ["feature_" + str(i) for i in range(0, data['data'].shape[1])]
	X = data['data']
	y = data['target']
	y[y != 1] = 0

	sss = StratifiedKFold(n_splits=3, shuffle=True)
	fold = 0
	for train_index, test_index in sss.split(X, y):
		fold += 1
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		clf_cc1 = AdaCC(n_estimators=250, algorithm='AdaCC1')
		clf_cc1.fit(X_train, y_train)

		clf_cc2 = AdaCC(n_estimators=250, algorithm='AdaCC2')
		clf_cc2.fit(X_train, y_train)

		print(dataset, fold, 'ada_cc1', calculate_performance(y_test, [clf_cc1.predict(X_test), clf_cc1.predict_proba(X_test)]))
		print(dataset, fold, 'ada_cc2', calculate_performance(y_test, [clf_cc2.predict(X_test), clf_cc2.predict_proba(X_test)]))
