import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns
sns.set()
sns.set_style("white")

data = h5py.File("../datasets/squalane_cn_dft.hdf5", "r")
ene_ref = -133.1

# Extracting the data
xyz = np.array(data.get("xyz"))
zs = np.array(data.get("zs"))
ene = np.array(data.get("ene"))*2625.5       # Converting the energy to kJ/mol
ene = ene - min(ene)
traj_idx = np.array(data.get("traj_idx"))
file_number = np.array(data.get("Filenumber"))

# Sorting the trajectories
idx_sorted = traj_idx.argsort()

ene = ene[idx_sorted]
traj_idx = traj_idx[idx_sorted]
file_number = file_number[idx_sorted]

n_traj = np.unique(traj_idx)

for item in n_traj:
    indices = np.where(traj_idx == item)

    idx_sorted = file_number[indices].argsort()

    ene[indices] = ene[indices][idx_sorted]
    traj_idx[indices] = traj_idx[indices][idx_sorted]
    file_number[indices] = file_number[indices][idx_sorted]


# Plotting
x = np.asarray(range(len(ene)))*0.5
print(len(x))
fig, ax = plt.subplots(1, figsize=(8,6))
ax.plot(x, ene)
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Energy (kJ/mol)")
# plt.savefig("squalane_dft.png", dpi=200)
plt.show()