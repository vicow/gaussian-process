import matplotlib.pyplot as plt
import pickle as cp
import numpy as np

x = np.linspace(0.025, 1, 40)

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

# Final learning curve

with open("data/vincent_accuracy_median.pkl", 'rb') as f:
    a_median = cp.load(f)
with open("data/vincent_accuracy_std.pkl", 'rb') as f:
    a_std = cp.load(f)
with open("data/accuracy_outlier_2.pkl", 'rb') as f:
    a2 = cp.load(f)
with open("data/accuracy_outlier_10.pkl", 'rb') as f:
    a3 = cp.load(f)
with open("data/accuracy_outlier_20.pkl", 'rb') as f:
    a4 = cp.load(f)
with open("data/gp_accuracy_outlier_2.pkl", 'rb') as f:
    a5 = cp.load(f)
with open("data/gp_accuracy_outlier_10.pkl", 'rb') as f:
    a6 = cp.load(f)
with open("data/gp_accuracy_outlier_20.pkl", 'rb') as f:
    a7 = cp.load(f)

# Vincent's data
a_median.append(100)
a_median = np.array(a_median) / 100
a_std.append(0)
a_std = np.array(a_std) / 100

#a_median = np.median(a, axis=0)
#a_std = np.std(a, axis=0)
a2_median = np.median(a2, axis=0)
a2_std = np.std(a2, axis=0)
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

plt.errorbar(x, a_median, a_std, label="Vincent")
plt.errorbar(x, a2_median, a2_std, label="Outliers > 2")
plt.errorbar(x, a3_median, a3_std, label="Outliers > 10")
plt.errorbar(x, a4_median, a4_std, label="Outliers > 20")
plt.errorbar(x, a5_median, a5_std, label="Outliers > 2 (GP)")
#plt.errorbar(x, a6_median, a6_std, label="Outliers > 10")
#plt.errorbar(x, a7_median, a7_std, label="Outliers > 20")
plt.xlabel("Relative time")
plt.ylabel("Accuracy")
plt.title("Learning curve using mixture of linear regressions with one feature")
#plt.title("Learning curve using gaussian processes with one feature")
plt.grid()
plt.legend(loc=4)
plt.ylim([plt.ylim()[0], 1.0])
plt.savefig("fig/accuracy_final.pdf")
plt.close()

#with open("data/mlr2_all_fixed_rmse.pkl", 'rb') as f:
#    r = cp.load(f)
with open("data/rmse_outlier_2.pkl", 'rb') as f:
    r2 = cp.load(f)
with open("data/rmse_outlier_10.pkl", 'rb') as f:
    r3 = cp.load(f)
with open("data/rmse_outlier_20.pkl", 'rb') as f:
    r4 = cp.load(f)
with open("data/gp_rmse_outlier_2.pkl", 'rb') as f:
    r5 = cp.load(f)
with open("data/gp_rmse_outlier_10.pkl", 'rb') as f:
    r6 = cp.load(f)
with open("data/gp_rmse_outlier_20.pkl", 'rb') as f:
    r7 = cp.load(f)

#r_median = np.median(r, axis=0)
#r_std = np.std(r, axis=0)
r2_median = np.median(r2, axis=0)
r2_std = np.std(r2, axis=0)
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

#plt.errorbar(x, r_median, r_std)
plt.errorbar(x, r2_median, r2_std, label="Outliers > 2")
plt.errorbar(x, r3_median, r3_std, label="Outliers > 10")
plt.errorbar(x, r4_median, r4_std, label="Outliers > 20")
plt.errorbar(x, r5_median, r5_std, label="Outliers > 2 (GP)")
#plt.errorbar(x, r6_median, r6_std, label="Outliers > 10")
#plt.errorbar(x, r7_median, r7_std, label="Outliers > 20")
plt.xlabel("Relative time")
plt.ylabel("RMSE")
plt.title("Learning curve using mixture of linear regressions with one feature")
#plt.title("Learning curve using gaussian processes with one feature")
plt.grid()
plt.legend(loc=1)
plt.ylim([0, plt.ylim()[1]])
plt.savefig("fig/rmse_final.pdf")
plt.close()