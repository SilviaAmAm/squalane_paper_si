from qml.aglaia.aglaia import ARMP
import tensorflow as tf
import h5py
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_context("poster")
sns.set_style("white")


def sort(traj_idx, fn, ene, xyz, zs):
    # Sorting the trajectories
    idx_sorted = traj_idx.argsort()

    ene = ene[idx_sorted]
    xyz = xyz[idx_sorted]
    zs = zs[idx_sorted]
    traj_idx = traj_idx[idx_sorted]
    fn = fn[idx_sorted]

    n_traj = np.unique(traj_idx)

    for item in n_traj:
        indices = np.where(traj_idx == item)

        idx_sorted = fn[indices].argsort()

        ene[indices] = ene[indices][idx_sorted]
        traj_idx[indices] = traj_idx[indices][idx_sorted]
        fn[indices] = fn[indices][idx_sorted]
        xyz[indices] = xyz[indices][idx_sorted]
        zs[indices] = zs[indices][idx_sorted]

    return traj_idx, fn, ene, xyz, zs


def remove_offset(reference, to_scale):
    """
    reference: list of energies
    to_scale: list of energies to scale
    return: list of scaled energies
    """

    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None])
        Y = tf.placeholder(tf.float32, [None])

        c_init = np.random.rand(1)
        c = tf.Variable(c_init, name='c', dtype=tf.float32)
        model = tf.subtract(X, c)
        cost_function = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(model - Y, 2)))
        optimiser = tf.train.GradientDescentOptimizer(0.0001).minimize(cost_function)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init)
        for i in range(100):
            sess.run(optimiser, feed_dict={X: to_scale, Y: reference})
            if i % 20 == 0:
                print("Step:", '%04d' % (i), "c=", sess.run(c))
        new_c = sess.run(c)[0]

    scaled_ene = to_scale - new_c
    return scaled_ene


if not os.path.exists("../outputs/compare_pm6_001/"):
    os.makedirs("../outputs/compare_pm6_001/")

# Squalane DFT data
data_dft = h5py.File("../datasets/squalane_cn_dft.hdf5", "r")

xyz_dft = np.array(data_dft.get("xyz"))
ene_dft = np.array(data_dft.get("ene")) * 2625.50
zs_dft = np.array(data_dft.get("zs"), dtype=np.int32)
n_atoms = len(zs_dft[0])
idx_dft = np.asarray(data_dft.get('traj_idx'), dtype=int)
fn_dft = np.asarray(data_dft.get('Filenumber'), dtype=int)

idx_dft, fn_dft, ene_dft, xyz_dft, zs_dft = sort(idx_dft, fn_dft, ene_dft, xyz_dft, zs_dft)
ref_ene = ene_dft[0]
ene_dft = ene_dft - ref_ene
print("DFT")
print(min(ene_dft), max(ene_dft))

# Squalane PM6 data
data_pm6 = h5py.File("../datasets/squalane_cn_pm6.hdf5", "r")

ene_pm6 = np.array(data_pm6.get("ene")) * 4.184
ene_pm6 = ene_pm6 - ene_pm6[0]

scaled_ene_pm6 = remove_offset(ene_dft, ene_pm6)

print("PM6")
print(np.mean(np.abs(ene_dft - scaled_ene_pm6)))

# Squalane NN data
if os.path.exists("../outputs/compare_pm6_001/NN_pred.npz"):
    data_NN = np.load("../outputs/compare_pm6_001/NN_pred.npz")
    ene_NN = data_NN["arr_0"]
else:
    acsf_hyperparameters = {"n_basis": 16, "r_min": 0.8, "r_cut": 3.0959454963762645, "tau": 1.7612032005732925}
    n_basis = acsf_hyperparameters["n_basis"]
    r_min = acsf_hyperparameters["r_min"]
    r_cut = acsf_hyperparameters["r_cut"]
    tau = acsf_hyperparameters["tau"]
    eta = 4 * np.log(tau) * ((n_basis - 1) / (r_cut - r_min)) ** 2
    zeta = - np.log(tau) / (2 * np.log(np.cos(np.pi / (4 * n_basis - 4))))
    acsf_params = {"nRs2": n_basis, "nRs3": n_basis, "nTs": n_basis, "rcut": r_cut, "acut": r_cut, "zeta": zeta,
                   "eta": eta}

    estimator = ARMP(representation_name='acsf', representation_params=acsf_params)
    estimator.load_nn("../outputs/hc6sq_train_001/saved_model")
    estimator.set_properties(ene_dft)
    estimator.generate_representation(xyz_dft, zs_dft, method='fortran')
    ene_NN = estimator.predict(list(range(0, len(ene_dft))))

    np.savez("../outputs/compare_pm6_001/NN_pred.npz", ene_NN)

scaled_ene_NN = remove_offset(ene_dft, ene_NN)
print(np.mean(np.abs(ene_dft - scaled_ene_NN)))

# Plotting the data

fig, ax = plt.subplots(figsize=(15, 8))

x = np.asarray(range(len(ene_dft))) / 2
ax.plot(x, ene_dft, label="DFT")
ax.plot(x, scaled_ene_NN, label="NN")
ax.plot(x, scaled_ene_pm6, label="PM6")
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Relative energy (kJ/mol)")
ax.legend()
plt.tight_layout()
# plt.savefig("comparison.png", dpi=150)
plt.show()
