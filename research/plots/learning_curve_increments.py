import matplotlib.pyplot as plt
import pickle as cp
import numpy as np

x = np.linspace(0.025, 1, 20)
x2 = np.linspace(0.025, 1, 40)
x3 = np.linspace(0.005, 1, 200)

outlier_threshold = 10000
granularity = 1

# Final learning curve

#with open("data/vincent_accuracy_median.pkl", 'rb') as f:
#    av_median = cp.load(f)
#with open("data/vincent_accuracy_std.pkl", 'rb') as f:
#    av_std = cp.load(f)
with open("data/vincent_reduced_dataset_accuracy_median.pkl", 'rb') as f:
    av2_median = cp.load(f)
with open("data/vincent_reduced_dataset_accuracy_std.pkl", 'rb') as f:
    av2_std = cp.load(f)
#with open("data/accuracy_outlier_2_granularity_0.001.pkl", 'rb') as f:
#    a2 = cp.load(f)
with open("data/accuracy_increments_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 1, False), 'rb') as f:
    a3 = cp.load(f)
with open("data/accuracy_increments_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 0.5, False), 'rb') as f:
    a4 = cp.load(f)
with open("data/accuracy_last_sample_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 1, False), 'rb') as f:
    a5 = cp.load(f)
with open("data/accuracy_increments_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 0.25, False), 'rb') as f:
    a6 = cp.load(f)
with open("data/accuracy_increments_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 0.1, False), 'rb') as f:
    a7 = cp.load(f)

# Vincent's data
#av_median.append(100)
#av_median = np.array(av_median) / 100
av2_median.append(100)
av2_median = np.array(av2_median) / 100
#a_median = a_median[np.arange(0, len(a_median), 2)]
#av_std.append(0)
#av_std = np.array(av_std) / 100
av2_std.append(0)
av2_std = np.array(av2_std) / 100
#a_std = a_std[np.arange(0, len(a_std), 2)]

#a_median = np.median(a, axis=0)
#a_std = np.std(a, axis=0)
#a2_median = np.median(a2, axis=0)
#a2_std = np.std(a2, axis=0)
a3_median = np.median(a3, axis=0)
a3_std = np.std(a3, axis=0)
a4_median = np.median(a4, axis=0)
a4_std = np.std(a4, axis=0)
a5_median = np.median(a5, axis=0)
a5_std = np.std(a5, axis=0)
a6_median = np.median(a6, axis=0)
a6_std = np.std(a6, axis=0)
a7_median = np.median(a7, axis=0)
a7_std = np.std(a7, axis=0)

print(len(x3), len(av2_median))

# plt.errorbar(x2, av_median, av_std, label="Vincent")
plt.errorbar(x2, av2_median, av2_std, label="Vincent (reduced)")
# plt.errorbar(x, a2_median, a2_std, label="Outliers > 2")
plt.errorbar(x2, a3_median, a3_std, label="Increments (1.0)")
plt.errorbar(x2, a4_median, a3_std, label="Increments (0.5)")
plt.errorbar(x2, a6_median, a3_std, label="Increments (0.25)")
plt.errorbar(x2, a7_median, a3_std, label="Increments (0.1)")
# plt.errorbar(x2, a4_median, a4_std, label="Increments (normalized)")
plt.errorbar(x2, a5_median, a5_std, label="Last sample")
#plt.errorbar(x, a6_median, a6_std, label="Outliers > 10")
#plt.errorbar(x, a7_median, a7_std, label="Outliers > 20")
plt.xlabel("Relative time")
plt.ylabel("Accuracy")
plt.title("Learning curve using linear regression")
#plt.title("Learning curve using gaussian processes with one feature")
plt.grid()
plt.legend(loc=4)
plt.ylim([plt.ylim()[0], 1.0])
plt.savefig("fig/increments_accuracy_outliers_%s_reduced_dataset_granularities.pdf" % outlier_threshold)
plt.close()

#with open("data/mlr2_all_fixed_rmse.pkl", 'rb') as f:
#    r = cp.load(f)
with open("data/rmse_increments_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 1, False), 'rb') as f:
    r3 = cp.load(f)
with open("data/rmse_increments_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 0.5, False), 'rb') as f:
    r4 = cp.load(f)
with open("data/rmse_last_sample_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 1, False), 'rb') as f:
    r5 = cp.load(f)
with open("data/rmse_increments_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 0.25, False), 'rb') as f:
    r6 = cp.load(f)
with open("data/rmse_increments_outlier_%s_granularity_%s_normalized_%s.pkl" % (outlier_threshold, 0.1, False), 'rb') as f:
    r7 = cp.load(f)

#r_median = np.median(r, axis=0)
#r_std = np.std(r, axis=0)
r3_median = np.median(r3, axis=0)
r3_std = np.std(r3, axis=0)
r4_median = np.median(r4, axis=0)
r4_std = np.std(r4, axis=0)
r5_median = np.median(r5, axis=0)
r5_std = np.std(r5, axis=0)
r6_median = np.median(r6, axis=0)
r6_std = np.std(r6, axis=0)
r7_median = np.median(r7, axis=0)
r7_std = np.std(r7, axis=0)

plt.errorbar(x2, r3_median, r3_std, label="Increments (1.0)")
plt.errorbar(x2, r4_median, r4_std, label="Increments (0.5)")
plt.errorbar(x2, r6_median, r3_std, label="Increments (0.25)")
plt.errorbar(x2, r7_median, r3_std, label="Increments (0.1)")
plt.errorbar(x2, r5_median, r5_std, label="Last sample")

plt.xlabel("Relative time")
plt.ylabel("RMSE")
plt.title("Learning curve using mixture of linear regressions with one feature")
#plt.title("Learning curve using gaussian processes with one feature")
plt.grid()
plt.legend(loc=1)
plt.ylim([0, plt.ylim()[1]])
plt.savefig("fig/increments_rmse_outliers_%s_reduced_dataset_granularities.pdf" % outlier_threshold)
plt.close()