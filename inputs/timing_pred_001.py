import pickle
import tensorflow as tf
from qml.qmlearn.preprocessing import AtomScaler
from qml.aglaia.aglaia import ARMP
import h5py
import glob
import numpy as np
import time
import os

# Create dir for the results
if not os.path.exists("../outputs/timings_pred_001/"):
    os.makedirs("../outputs/timings_pred_001/")

# Squalane data
data_squal = h5py.File("../datasets/squalane_cn_dft.hdf5", "r")
ref_ene = -133.1 * 2625.50

xyz_squal = np.array(data_squal.get("xyz"))
ene_squal = np.array(data_squal.get("ene")) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)
n_atoms_squal = len(zs_squal[0])

# Loading the scaler and scaling the energy
scaling = pickle.load(open("../outputs/make_scaler_001/scaler.pickle", "rb"))
ene_scaled = scaling.transform(zs_squal, ene_squal)

model_paths = ["../outputs/hc%isq_train_001/saved_model" % i for i in range(1, 7)]
models_rep_time = []
models_pred_time = []

# Making a dictionary of all the hyper-parameters:
acsf_hyperparameters = {}
acsf_hyperparameters["hc1_sq"] = {"n_basis": 15, "r_min": 0.8, "r_cut": 4.333268208108573, "tau": 1.714566950825575}
acsf_hyperparameters["hc2_sq"] = {"n_basis": 19, "r_min": 0.8, "r_cut": 3.410196656676407, "tau": 1.4169727865082424}
acsf_hyperparameters["hc3_sq"] = {"n_basis": 13, "r_min": 0.8, "r_cut": 3.5901786999505747, "tau": 1.7480712406910572}
acsf_hyperparameters["hc4_sq"] = {"n_basis": 13, "r_min": 0.8, "r_cut": 3.440665282995163, "tau": 1.8930017259735163}
acsf_hyperparameters["hc5_sq"] = {"n_basis": 12, "r_min": 0.8, "r_cut": 3.7003886787730114, "tau": 2.2194019378678367}
acsf_hyperparameters["hc6_sq"] = {"n_basis": 16, "r_min": 0.8, "r_cut": 3.0959454963762645, "tau": 1.7612032005732925}

for counter, model_path in enumerate(model_paths):
    rep_time = []
    pred_time = []
    for _ in range(5):
        acsf_hp = acsf_hyperparameters[f"hc{counter+1}_sq"]

        # ACSF parameters
        n_basis = acsf_hp["n_basis"]
        r_min = acsf_hp["r_min"]
        r_cut = acsf_hp["r_cut"]
        tau = acsf_hp["tau"]
        eta = 4 * np.log(tau) * ((n_basis - 1) / (r_cut - r_min)) ** 2
        zeta = - np.log(tau) / (2 * np.log(np.cos(np.pi / (4 * n_basis - 4))))

        acsf_params = {"nRs2": n_basis, "nRs3": n_basis, "nTs": n_basis, "rcut": r_cut, "acut": r_cut, "zeta": zeta,
                       "eta": eta}

        # Load the estimator
        estimator = ARMP(representation_name='acsf', representation_params=acsf_params)
        estimator.load_nn(model_path)

        # Give the data to the model and make the representations
        estimator.set_properties(ene_scaled)
        start = time.time()
        estimator.generate_representation(xyz_squal, zs_squal, method='fortran')
        end = time.time()
        rep_time.append(end - start)

        # Predict squalane
        pred_idx_squal = list(range(0, len(ene_squal)))
        start = time.time()
        pred_squal = estimator.predict(pred_idx_squal)
        end = time.time()
        pred_time.append(end - start)

        tf.reset_default_graph()
        del estimator

    print("For model hc%i_sq, representation time: %.2f pm %.2f, prediction time: %.2f pm %.2f" % (
    counter+1, np.mean(rep_time), np.std(rep_time), np.mean(pred_time), np.std(pred_time)))

    models_rep_time.append(rep_time)
    models_pred_time.append(pred_time)

np.savez("../outputs/timings_pred_001/timings.npz", np.asarray(models_rep_time), np.asarray(models_pred_time))

