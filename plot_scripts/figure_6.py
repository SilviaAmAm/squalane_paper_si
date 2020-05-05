import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
sns.set()
sns.set_context("talk")
sns.set_style("white")

import numpy as np


data = np.load("../outputs/hc2sq_predict_001/sorted_predictions.npz")

pred_ene_squal = data["arr_4"]
true_ene_squal = data["arr_5"]

r2 = r2_score(true_ene_squal, pred_ene_squal)

print("Squalane error for MAE: %s kJ/mol" % str(np.std(true_ene_squal-pred_ene_squal)))
print("Squalane R2: %s " % str(r2))

x = np.asarray(range(len(pred_ene_squal)))
x = (x/2).astype(np.int)
fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(15, 5))

ax0.plot(x, true_ene_squal, ms=10, label="DFT")
ax0.plot(x, pred_ene_squal, ms=10, label="NN")
ax0.set_xlabel("Time (fs)")
ax0.set_ylabel("Energy (kJ/mol)")
ax0.set_ylim((-180, 180))
ax0.set_aspect(1)
ax0.legend()

ax1.scatter(true_ene_squal, pred_ene_squal, s=10, c=sns.color_palette()[2])
ax1.set_xlabel("DFT energy (kJ/mol)")
ax1.set_ylabel("NN energy (kJ/mol)")
ax1.text(-160, 150, f"R2 = {r2:.2f}")
ax1.set_xlim((-180, 180))
ax1.set_ylim((-180, 180))
ax1.set_aspect(1)

plt.tight_layout()
# plt.savefig("hc2pred.png", dpi=200)
plt.show()