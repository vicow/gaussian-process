import matplotlib.pyplot as plt
import pickle as cp
import numpy as np

x = np.linspace(0.025, 1, 40)

# Single learning curve

with open("accuracy_run.pkl", 'rb') as f:
    a = cp.load(f)

a = np.array(a)
a = a[0, :] * 100

plt.plot(x, a, '-o')
plt.xlabel("Relative time")
plt.ylabel("Accuracy")
plt.title("Accuracy on test set for mixture of 2 linear regressions")
plt.savefig('accuracy_run.pdf')
plt.close()

with open("rmse_run.pkl", 'rb') as f:
    r = cp.load(f)

r = np.array(r)[0]
plt.plot(x, r, '-o')
plt.xlabel("Relative time")
plt.ylabel("RMSE")
plt.title("RMSE on test set for mixture of 2 linear regressions")
plt.savefig('rmse_run.pdf')
plt.close()

# Final learning curve

with open("accuracy_all.pkl", 'rb') as f:
    a = cp.load(f)

a_median = np.median(a, axis=0)
a_std = np.std(a, axis=0)

plt.errorbar(x, a_median, a_std)
plt.xlabel("Relative time")
plt.ylabel("Accuracy")
plt.title("Learning curve for mixture of 2 linear regression")
plt.savefig("accuracy_all.pdf")
plt.close()

with open("rmse_all.pkl", 'rb') as f:
    r = cp.load(f)

r_median = np.median(r, axis=0)
r_std = np.std(r, axis=0)
print(a_std)

plt.errorbar(x, r_median, r_std)
plt.xlabel("Relative time")
plt.ylabel("RMSE")
plt.title("Learning curve for mixture of 2 linear regression")
plt.savefig("rmse_all.pdf")
plt.close()