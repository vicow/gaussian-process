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

with open("mlr2_accuracy.pkl", 'rb') as f:
    a = cp.load(f)

with open("mlr2_all_fixed_accuracy.pkl", 'rb') as f:
    a2 = cp.load(f)

a_median = np.median(a, axis=0)
a_std = np.std(a, axis=0)
a2_median = np.median(a2, axis=0)
a2_std = np.std(a2, axis=0)

plt.errorbar(x, a_median, a_std)
plt.errorbar(x, a2_median, a2_std)
plt.xlabel("Relative time")
plt.ylabel("Accuracy")
plt.title("Learning curve for mixture of 2 linear regression")
plt.grid()
plt.legend(["Fixed parameters", "Learned parameters"])
plt.savefig("accuracy.pdf")
plt.close()

with open("mlr2_rmse.pkl", 'rb') as f:
    r = cp.load(f)
with open("mlr2_all_fixed_rmse.pkl", 'rb') as f:
    r2 = cp.load(f)

r_median = np.median(r, axis=0)
r_std = np.std(r, axis=0)
r2_median = np.median(r2, axis=0)
r2_std = np.std(r2, axis=0)

plt.errorbar(x, r_median, r_std)
plt.errorbar(x, r2_median, r2_std)
plt.xlabel("Relative time")
plt.ylabel("RMSE")
plt.title("Learning curve for mixture of 2 linear regression")
plt.grid()
plt.legend(["Fixed parameters", "Learned parameters"])
plt.savefig("rmse.pdf")
plt.close()