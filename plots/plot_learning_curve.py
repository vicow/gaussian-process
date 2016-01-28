
import matplotlib.pyplot as plt
import pickle as cp
import numpy as np

x = np.linspace(0.025, 1, 40)
outlier_threshol = 2

# Final learning curve

with open("data/accuracy_outlier_%s.pkl" % outlier_threshol, 'rb') as f:
    a = cp.load(f)

a_median = np.median(a, axis=0)
a_std = np.std(a, axis=0)

plt.errorbar(x, a_median, a_std)
plt.xlabel("Relative time")
plt.ylabel("Accuracy")
plt.title("Learning curve for mixture of 2 linear regressions with one feature")
plt.ylim([plt.ylim()[0], 1.0])
plt.grid()
plt.savefig("fig/accuracy_outlier_%s.pdf" % outlier_threshol)
plt.close()

with open("data/rmse_outlier_%s.pkl" % outlier_threshol, 'rb') as f:
    r = cp.load(f)

r_median = np.median(r, axis=0)
r_std = np.std(r, axis=0)

with open("data/rmse_failed_outlier_%s.pkl" % outlier_threshol, 'rb') as f:
    rf = cp.load(f)

rf_median = np.median(rf, axis=0)
rf_std = np.std(rf, axis=0)


with open("data/rmse_success_outlier_%s.pkl" % outlier_threshol, 'rb') as f:
    rs = cp.load(f)

rs_median = np.median(rs, axis=0)
rs_std = np.std(rs, axis=0)

plt.errorbar(x, r_median, r_std, label="Total")
plt.errorbar(x, rf_median, rf_std, label="Failed")
plt.errorbar(x, rs_median, rs_std, label="Successful")
plt.xlabel("Relative time")
plt.ylabel("RMSE")
plt.title("Learning curve for mixture of 2 linear regressions with one feature")
plt.ylim([0, plt.ylim()[1]])
plt.grid()
plt.legend()
plt.savefig("fig/rmse_outlier_%s.pdf" % outlier_threshol)
plt.close()