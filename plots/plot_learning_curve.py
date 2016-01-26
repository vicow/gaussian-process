
import matplotlib.pyplot as plt
import pickle as cp
import numpy as np

x = np.linspace(0.025, 1, 40)

# Final learning curve

with open("accuracy_all.pkl", 'rb') as f:
    a = cp.load(f)

a_median = np.median(a, axis=0)
a_std = np.std(a, axis=0)

plt.errorbar(x, a_median, a_std)
plt.xlabel("Relative time")
plt.ylabel("Accuracy")
plt.title("Learning curve for mixture of 3 linear regressions with one feature")
plt.grid()
plt.savefig("mlr3_accuracy.pdf")
plt.close()

with open("rmse_all.pkl", 'rb') as f:
    r = cp.load(f)

r_median = np.median(r, axis=0)
r_std = np.std(r, axis=0)

plt.errorbar(x, r_median, r_std)
plt.xlabel("Relative time")
plt.ylabel("RMSE")
plt.title("Learning curve for mixture of 3 linear regressions with one feature")
plt.grid()
plt.savefig("mlr3_rmse.pdf")
plt.close()