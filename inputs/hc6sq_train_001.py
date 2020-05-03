from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import pickle
from random import shuffle
import os

# Creating output dir
if not os.path.exists("../outputs/hc6sq_train_001/"):
    os.makedirs("../outputs/hc6sq_train_001/")

# Getting the dataset
data_methane = h5py.File("../datasets/methane_cn_dft.hdf5", "r")
data_isopentane = h5py.File("../datasets/isopentane_cn_dft.hdf5", "r")
data_squal = h5py.File("../datasets/squalane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("../datasets/2isohexane_cn_dft.hdf5")
data_3isohex = h5py.File("../datasets/3isohexane_cn_dft.hdf5")

ref_ene = -133.1 * 2625.50

n_samples_methane = 8000
n_samples_isopentane = 4000
n_samples_2isohex = 1500
n_samples_3isohex = 1500

zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)
n_atoms_squal = len(zs_squal[0])
print("The number of squalane samples is: %i" % len(zs_squal))

# Data for methane
traj_idx_methane = np.asarray(data_methane.get('traj_idx'), dtype=int)

idx_train_methane = np.where(traj_idx_methane != 14)[0]
shuffle(idx_train_methane)
idx_train_methane = idx_train_methane[:n_samples_methane]

print("The number of methane samples is: %i (train)" % (len(idx_train_methane)))

xyz_methane = np.array(data_methane.get("xyz"))[idx_train_methane]
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)[idx_train_methane]
ene_methane = np.array(data_methane.get("ene"))[idx_train_methane]* 2625.50 - ref_ene

pad_xyz_methane = np.concatenate((xyz_methane, np.zeros((xyz_methane.shape[0], n_atoms_squal - xyz_methane.shape[1], 3))), axis=1)
pad_zs_methane = np.concatenate((zs_methane, np.zeros((zs_methane.shape[0], n_atoms_squal - zs_methane.shape[1]), dtype=np.int32)), axis=1)

# Data for isopentane
traj_idx_isopentane = np.asarray(data_isopentane.get('traj_idx'), dtype=int)

idx_train_isopentane = np.where(traj_idx_isopentane != 22)[0]
shuffle(idx_train_isopentane)
idx_train_isopentane = idx_train_isopentane[:n_samples_isopentane]

print("The number of isopentane samples is: %i (train)" % (len(idx_train_isopentane)))

xyz_isopentane = np.array(data_isopentane.get("xyz"))[idx_train_isopentane]
zs_isopentane = np.array(data_isopentane.get("zs"), dtype=np.int32)[idx_train_isopentane]
ene_isopentane = np.array(data_isopentane.get("ene"))[idx_train_isopentane]* 2625.50 - ref_ene

pad_xyz_isopentane = np.concatenate((xyz_isopentane, np.zeros((xyz_isopentane.shape[0], n_atoms_squal - xyz_isopentane.shape[1], 3))), axis=1)
pad_zs_isopentane = np.concatenate((zs_isopentane, np.zeros((zs_isopentane.shape[0], n_atoms_squal - zs_isopentane.shape[1]), dtype=np.int32)), axis=1)

# Data for 2-isohexane
traj_idx_2isohex = np.asarray(data_2isohex.get('traj_idx'), dtype=int)

idx_train_2isohex = np.where(traj_idx_2isohex != 12)[0]
shuffle(idx_train_2isohex)
idx_train_2isohex = idx_train_2isohex[:n_samples_2isohex]

print("The number of 2-isohexane samples is: %i (train)" % (len(idx_train_2isohex)))

xyz_2isohex = np.array(data_2isohex.get("xyz"))[idx_train_2isohex]
zs_2isohex = np.array(data_2isohex.get("zs"), dtype=np.int32)[idx_train_2isohex]
ene_2isohex = np.array(data_2isohex.get("ene"))[idx_train_2isohex]* 2625.50 - ref_ene

pad_xyz_2isohex = np.concatenate((xyz_2isohex, np.zeros((xyz_2isohex.shape[0], n_atoms_squal - xyz_2isohex.shape[1], 3))), axis=1)
pad_zs_2isohex = np.concatenate((zs_2isohex, np.zeros((zs_2isohex.shape[0], n_atoms_squal - zs_2isohex.shape[1]), dtype=np.int32)), axis=1)

# Data for 3-isohexane
traj_idx_3isohex = np.asarray(data_3isohex.get('traj_idx'), dtype=int)

idx_train_3isohex = np.where(traj_idx_3isohex != 14)[0]
shuffle(idx_train_3isohex)
idx_train_3isohex = idx_train_3isohex[:n_samples_3isohex]

print("The number of 3-isohexane samples is: %i (train)" % (len(idx_train_3isohex)))

xyz_3isohex = np.array(data_3isohex.get("xyz"))[idx_train_3isohex]
zs_3isohex = np.array(data_3isohex.get("zs"), dtype=np.int32)[idx_train_3isohex]
ene_3isohex = np.array(data_3isohex.get("ene"))[idx_train_3isohex]* 2625.50 - ref_ene

pad_xyz_3isohex = np.concatenate((xyz_3isohex, np.zeros((xyz_3isohex.shape[0], n_atoms_squal - xyz_3isohex.shape[1], 3))), axis=1)
pad_zs_3isohex = np.concatenate((zs_3isohex, np.zeros((zs_3isohex.shape[0], n_atoms_squal - zs_3isohex.shape[1]), dtype=np.int32)), axis=1)


# Concatenating all the data
concat_xyz = np.concatenate((pad_xyz_methane, pad_xyz_isopentane, pad_xyz_2isohex, pad_xyz_3isohex))
concat_ene = np.concatenate((ene_methane, ene_isopentane, ene_2isohex, ene_3isohex))
concat_zs = np.concatenate((pad_zs_methane, pad_zs_isopentane, pad_zs_2isohex, pad_zs_3isohex))

zs_for_scaler = list(zs_methane) + list(zs_isopentane) + list(zs_2isohex) + list(zs_3isohex)

scaling = pickle.load(open("../outputs/make_scaler_001/scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler, concat_ene)

# ACSF parameters
n_basis = 16
r_min = 0.8
r_cut = 3.0959454963762645
tau = 1.7612032005732925
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

# Generate estimator
estimator = ARMP(iterations=900, l1_reg=0.00018891702136509527, l2_reg=2.172308772374847e-08, learning_rate=0.001471842348676605, representation_name='acsf', representation_params=acsf_params, tensorboard=True, store_frequency=10, tensorboard_subdir="../outputs/hc6sq_train_001/tensorboard", hidden_layer_sizes=(62,142,), batch_size=23)

estimator.set_properties(concat_ene_scaled)
estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

# Training and testing
idx_training = list(range(len(concat_ene_scaled)))
shuffle(idx_training)

estimator.fit(idx_training)

estimator.save_nn("../outputs/hc6sq_train_001/saved_model")
