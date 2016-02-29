import matplotlib.pyplot as plt
import pickle as cp
import numpy as np

x = np.linspace(0.025, 1, 20)
x2 = np.linspace(0.025, 1, 40)

# Single learning curve
#
# with open("accuracy_all.pkl", 'rb') as f:
#     a = cp.load(f)
#
# a = np.array(a)[0]
#
# plt.plot(x, a, '-o')
# plt.xlabel("Relative time")
# plt.ylabel("Accuracy")
# plt.title("Accuracy on test set for mixture of 2 linear regressions")
# plt.savefig('accuracy_run.pdf')
# plt.close()
#
# with open("rmse_all.pkl", 'rb') as f:
#     r = cp.load(f)
#
# r = np.array(r)[0]
# plt.plot(x, r, '-o')
# plt.xlabel("Relative time")
# plt.ylabel("RMSE")
# plt.title("RMSE on test set for mixture of 2 linear regressions")
# plt.savefig('rmse_run.pdf')
# plt.close()

outlier_threshold = 10000000

# Final learning curve

with open("data/vincent_accuracy_median.pkl", 'rb') as f:
    a_median = cp.load(f)
with open("data/vincent_accuracy_std.pkl", 'rb') as f:
    a_std = cp.load(f)
with open("data/accuracy_outlier_2_granularity_0.001.pkl", 'rb') as f:
    a2 = cp.load(f)
with open("data/accuracy_increments_outlier_%s_granularity_1.pkl" % (outlier_threshold,), 'rb') as f:
    a3 = cp.load(f)

# Vincent's data
a_median.append(100)
a_median = np.array(a_median) / 100
#a_median = a_median[np.arange(0, len(a_median), 2)]
a_std.append(0)
a_std = np.array(a_std) / 100
#a_std = a_std[np.arange(0, len(a_std), 2)]

#a_median = np.median(a, axis=0)
#a_std = np.std(a, axis=0)
a2_median = np.median(a2, axis=0)
a2_std = np.std(a2, axis=0)
a3_median = np.median(a3, axis=0)
a3_std = np.std(a3, axis=0)

plt.errorbar(x2, a_median, a_std, label="Vincent")
plt.errorbar(x, a2_median, a2_std, label="Outliers > 2")
plt.errorbar(x2, a3_median, a3_std, label="Increments")
#plt.errorbar(x, a6_median, a6_std, label="Outliers > 10")
#plt.errorbar(x, a7_median, a7_std, label="Outliers > 20")
plt.xlabel("Relative time")
plt.ylabel("Accuracy")
plt.title("Learning curve using mixture of linear regressions with one feature")
#plt.title("Learning curve using gaussian processes with one feature")
plt.grid()
plt.legend(loc=4)
plt.ylim([plt.ylim()[0], 1.0])
plt.savefig("fig/increments_accuracy_outliers_%s.pdf" % outlier_threshold)
plt.close()

#with open("data/mlr2_all_fixed_rmse.pkl", 'rb') as f:
#    r = cp.load(f)
with open("data/rmse_outlier_2_granularity_0.001.pkl", 'rb') as f:
    r2 = cp.load(f)
with open("data/rmse_increments_outlier_%s_granularity_1.pkl" % (outlier_threshold,), 'rb') as f:
    r3 = cp.load(f)

#r_median = np.median(r, axis=0)
#r_std = np.std(r, axis=0)
r2_median = np.median(r2, axis=0)
r2_std = np.std(r2, axis=0)
r3_median = np.median(r3, axis=0)
r3_std = np.std(r3, axis=0)

#plt.errorbar(x, r_median, r_std)
plt.errorbar(x, r2_median, r2_std, label="Outliers > 2")
plt.errorbar(x2, r3_median, r3_std, label="Increments")
#plt.errorbar(x, r6_median, r6_std, label="Outliers > 10")
#plt.errorbar(x, r7_median, r7_std, label="Outliers > 20")
plt.xlabel("Relative time")
plt.ylabel("RMSE")
plt.title("Learning curve using mixture of linear regressions with one feature")
#plt.title("Learning curve using gaussian processes with one feature")
plt.grid()
plt.legend(loc=1)
plt.ylim([0, plt.ylim()[1]])
plt.savefig("fig/increments_rmse.pdf")
plt.close()