import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
sns.set()
sns.set_context("talk")
sns.set_style("white")


def plot_non_zoomed(concat_ene, concat_ene_scaled):
    fig, ax = plt.subplots(1, figsize=(16, 6))
    ax.scatter(list(range(len(concat_ene))), concat_ene, label="Non scaled", s=20)
    ax.scatter(list(range(len(concat_ene))), concat_ene_scaled, label="Scaled", s=20)
    hydrocarbons = ["Methane", "Ethane", "Isobutane", "Isopentane", "2-Isohexane", "3-Isohexane", "Squalane"]

    for i in range(5000,5000*(len(hydrocarbons)+1), 5000):
        if i == 5000*len(hydrocarbons):
            ax.text(i - 2500, 3e5, hydrocarbons[int((i - 1) / 5000)], size=15, ha='center', va='center')
            break
        ax.axvline(x=i, color="black", linestyle='--')
        ax.text(i-2500, 3e5, hydrocarbons[int((i-1)/5000)], size=15, ha='center', va='center')

    ax.legend()
    ax.set_xlim((-10, 5000 * len(hydrocarbons)))
    ax.set_ylim((-3.25*1e6,0.5*1e6))
    ax.set_xlabel("Frames")
    ax.set_ylabel("Energy (kJ/mol)")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    # plt.savefig("non_zoomed_trajectories.png", dpi=150)
    plt.show()


def plot_zoomed(concat_ene, concat_ene_scaled):
    fig, ax = plt.subplots(1, figsize=(16, 6))
    ax.scatter(list(range(len(concat_ene))), concat_ene, label="Non scaled", s=20)
    ax.scatter(list(range(len(concat_ene))), concat_ene_scaled, label="Scaled", s=20)
    hydrocarbons = ["Methane", "Ethane", "Isobutane", "Isopentane", "2-Isohexane", "3-Isohexane", "Squalane"]

    offset = -1.5e5

    for i in range(5000, 5000 * (len(hydrocarbons) + 1), 5000):
        ax.axvline(x=i, color="black", linestyle='--')
        if i == 5000 * len(hydrocarbons):
            ax.text(i - 4000, concat_ene[i - 5000] + offset, hydrocarbons[int((i - 1) / 5000)], ha='center',
                    va='center')
        else:
            ax.text(i - 2500, concat_ene[i - 5000] + offset, hydrocarbons[int((i - 1) / 5000)], ha='center',
                    va='center')

    ax.legend()
    ax.set_xlim((-10, 5000 * len(hydrocarbons)))
    ax.set_ylim((-2e2, 2e2))
    ax.set_xlabel("Frames")
    ax.set_ylabel("Energy (kJ/mol)")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    # plt.savefig("zoomed_trajectories.png", dpi=150)
    plt.show()


data_methane = h5py.File("../datasets/methane_cn_dft.hdf5", "r")
data_ethane = h5py.File("../datasets/ethane_cn_dft.hdf5", "r")
data_isobutane = h5py.File("../datasets/isobutane_cn_dft.hdf5", "r")
data_isopentane = h5py.File("../datasets/isopentane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("../datasets/2isohexane_cn_dft.hdf5", "r")
data_3isohex = h5py.File("../datasets/3isohexane_cn_dft.hdf5", "r")
data_squal = h5py.File("../datasets/squalane_cn_dft.hdf5", "r")

ref_ene = -133.1 * 2625.50

n_samples = 5000

ene_squal = np.array(data_squal.get("ene")[:n_samples]) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squal.get("zs")[:n_samples], dtype=np.int32)

ene_3hex = np.array(data_3isohex.get("ene")[:n_samples]) * 2625.50
ene_3hex = ene_3hex - ref_ene
zs_3hex = np.array(data_3isohex.get("zs")[:n_samples], dtype=np.int32)

ene_2hex = np.array(data_2isohex.get("ene")[:n_samples]) * 2625.50
ene_2hex = ene_2hex - ref_ene
zs_2hex = np.array(data_2isohex.get("zs")[:n_samples], dtype=np.int32)

ene_isopent = np.array(data_isopentane.get("ene")[:n_samples]) * 2625.50
ene_isopent = ene_isopent - ref_ene
zs_isopent = np.array(data_isopentane.get("zs")[:n_samples], dtype=np.int32)

ene_isobutane = np.array(data_isobutane.get("ene")[:n_samples]) * 2625.50
ene_isobutane = ene_isobutane - ref_ene
zs_isobutane = np.array(data_isobutane.get("zs")[:n_samples], dtype=np.int32)

ene_ethane = np.array(data_ethane.get("ene")[:n_samples]) * 2625.50
ene_ethane = ene_ethane - ref_ene
zs_ethane = np.array(data_ethane.get("zs")[:n_samples], dtype=np.int32)

ene_methane = np.array(data_methane.get("ene")[:n_samples])[::-1] * 2625.50
ene_methane = ene_methane - ref_ene
zs_methane = np.array(data_methane.get("zs")[:n_samples], dtype=np.int32)[::-1]

zs_for_scaler_long = list(zs_methane) + list(zs_ethane) + list(zs_isobutane) + list(zs_isopent) + list(zs_2hex) + list(zs_3hex) + list(zs_squal)
concat_ene = np.concatenate((ene_methane, ene_ethane, ene_isobutane, ene_isopent, ene_2hex, ene_3hex, ene_squal))

scaling = pickle.load(open("../outputs/make_scaler_001/scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler_long, concat_ene)

plot_non_zoomed(concat_ene, concat_ene_scaled)

plot_zoomed(concat_ene, concat_ene_scaled)
