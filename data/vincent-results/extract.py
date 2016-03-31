import json
import math
import cPickle as cp
import numpy as np

accuracy = []
std = []

for p in np.linspace(2.5, 100, 40):
    if p == 100.0:
        break
    file_name = "../vincent/reduced_dataset/results_C=100.0_action=predict_combiner=svm_dosave=False_gamma=0.1_n=10_nobaseline=False_p=%s_paper=False_proba=True_talk=False_which=all.txt" % p
    #file_name = "results_C=100.0_action=predict_combiner=svm_gamma=0.1_n=10_nobaseline=False_p=%s_paper=False_proba=True_which=all.txt" % p
    with open(file_name, "r") as f:
        s = f.read()
        d = json.loads(s)
        accuracy.append(d["median"])
        std.append(math.sqrt(d["variance"]))

with open("vincent_reduced_dataset_accuracy_median.pkl", 'wb') as f:
    cp.dump(accuracy, f)

with open("vincent_reduced_dataset_accuracy_std.pkl", 'wb') as f:
    cp.dump(std, f)