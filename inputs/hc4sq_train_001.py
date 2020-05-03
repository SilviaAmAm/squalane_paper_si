from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import pickle
from random import shuffle
import os

# Creating output dir
if not os.path.exists("../outputs/hc4sq_train_001/"):
    os.makedirs("../outputs/hc4sq_train_001/")

# Getting the dataset
data_methane = h5py.File("../datasets/methane_cn_dft.hdf5", "r")
data_ethane = h5py.File("../datasets/ethane_cn_dft.hdf5")
data_isobutane = h5py.File("../datasets/isobutane_cn_dft.hdf5")
data_isopentane = h5py.File("../datasets/isopentane_cn_dft.hdf5", "r")
data_squal = h5py.File("../datasets/squalane_cn_dft.hdf5", "r")

ref_ene = -133.1 * 2625.50

n_samples_methane = 7500
n_samples_ethane = 3500
n_samples_isobutane = 2500
n_samples_isopentane = 1500

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

# Data for ethane
traj_idx_ethane = np.asarray(data_ethane.get('traj_idx'), dtype=int)

idx_train_ethane = np.where(traj_idx_ethane != 7)[0]
shuffle(idx_train_ethane)
idx_train_ethane = idx_train_ethane[:n_samples_ethane]

print("The number of ethane samples is: %i (train)" % (len(idx_train_ethane)))

xyz_ethane = np.array(data_ethane.get("xyz"))[idx_train_ethane]
zs_ethane = np.array(data_ethane.get("zs"), dtype=np.int32)[idx_train_ethane]
ene_ethane = np.array(data_ethane.get("ene"))[idx_train_ethane]* 2625.50 - ref_ene

pad_xyz_ethane = np.concatenate((xyz_ethane, np.zeros((xyz_ethane.shape[0], n_atoms_squal - xyz_ethane.shape[1], 3))), axis=1)
pad_zs_ethane = np.concatenate((zs_ethane, np.zeros((zs_ethane.shape[0], n_atoms_squal - zs_ethane.shape[1]), dtype=np.int32)), axis=1)

# Data for isobutane
traj_idx_isobutane = np.asarray(data_isobutane.get('traj_idx'), dtype=int)

idx_train_isobutane = np.where(traj_idx_isobutane != 2)[0]
shuffle(idx_train_isobutane)
idx_train_isobutane = idx_train_isobutane[:n_samples_isobutane]

print("The number of isobutane samples is: %i (train)" % (len(idx_train_isobutane)))

xyz_isobutane = np.array(data_isobutane.get("xyz"))[idx_train_isobutane]
zs_isobutane = np.array(data_isobutane.get("zs"), dtype=np.int32)[idx_train_isobutane]
ene_isobutane = np.array(data_isobutane.get("ene"))[idx_train_isobutane]* 2625.50 - ref_ene

pad_xyz_isobutane = np.concatenate((xyz_isobutane, np.zeros((xyz_isobutane.shape[0], n_atoms_squal - xyz_isobutane.shape[1], 3))), axis=1)
pad_zs_isobutane = np.concatenate((zs_isobutane, np.zeros((zs_isobutane.shape[0], n_atoms_squal - zs_isobutane.shape[1]), dtype=np.int32)), axis=1)

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

# Concatenating all the data
concat_xyz = np.concatenate((pad_xyz_methane, pad_xyz_ethane, pad_xyz_isobutane, pad_xyz_isopentane))
concat_ene = np.concatenate((ene_methane, ene_ethane, ene_isobutane, ene_isopentane))
concat_zs = np.concatenate((pad_zs_methane, pad_zs_ethane, pad_zs_isobutane, pad_zs_isopentane))

zs_for_scaler = list(zs_methane) + list(zs_ethane) + list(zs_isobutane) + list(zs_isopentane)

scaling = pickle.load(open("../outputs/make_scaler_001/scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler, concat_ene)

# ACSF parameters
n_basis = 13
r_min = 0.8
r_cut = 3.440665282995163
tau = 1.8930017259735163
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

# Generate estimator
estimator = ARMP(iterations=1181, l1_reg=0.00025021816931208597, l2_reg=4.0694790725485916e-05, learning_rate=0.0007055546397595542, representation_name='acsf', representation_params=acsf_params, tensorboard=True, store_frequency=10, tensorboard_subdir="../outputs/hc4sq_train_001/tensorboard", hidden_layer_sizes=(94,174,), batch_size=26)

estimator.set_properties(concat_ene_scaled)
estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

# Training and testing
idx_training = list(range(len(concat_ene_scaled)))
shuffle(idx_training)

estimator.fit(idx_training)

estimator.save_nn("../outputs/hc4sq_train_001/saved_model")
