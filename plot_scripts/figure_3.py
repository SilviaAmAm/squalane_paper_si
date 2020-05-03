import numpy as np
import bisect
import glob
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("white")

import h5py

def get_h_c_abstracted(traj_id, xyz, zs):

    h_id = np.zeros(traj_id.shape, dtype=np.int32)
    c_id = np.zeros(traj_id.shape, dtype=np.int32)

    unique_traj_idx = np.unique(traj_id)

    for id in unique_traj_idx:
        # Frames that belong to a particular trajectory
        idx_frames = np.where(traj_id==id)[0]

        # Coordinates of those frames
        xyz_frames = xyz[idx_frames]
        zs_frames = zs[idx_frames]

        # f = open("vmdtraj.xyz", "w")
        # idx_to_char = {1: "H", 6:"C", 7:"N"}
        # for traj_frame in range(len(zs_frames)):
        #     f.write(str(len(zs_frames[traj_frame])))
        #     f.write("\n\n")
        #
        #     for n in range(len(zs_frames[traj_frame])):
        #         f.write(str(idx_to_char[zs_frames[traj_frame][n]]))
        #         f.write("\t")
        #         for i in range(3):
        #             f.write(str(xyz_frames[traj_frame][n][i]))
        #             f.write("\t")
        #         f.write("\n")
        # f.close()
        # exit()

        # Coordinates of the last frame of a trajectory
        xyz_last_frame = xyz_frames[-1]
        zs_last_frame = zs_frames[-1]

        # Coordinates of the first frame of a trajectory
        xyz_first_frame = xyz_frames[0]
        zs_first_frame = zs_frames[0]

        # The H being abstracted will be the one with the shortest distance to the CN carbon
        idx_of_all_h = np.where(zs_last_frame == 1)[0]
        min_dist = 10
        abstracted_h = -1       # Index of the abstracted hydrogen
        for h in idx_of_all_h:
            dist = np.linalg.norm((xyz_last_frame[h]- xyz_last_frame[-2]))    # distance between a H and the CN carbon
            if dist < min_dist:
                min_dist = dist
                abstracted_h = int(h)

        h_id[idx_frames] = abstracted_h

        # The C initially bonded to the abstracted H will be the one with the shortest distance to it in the first frame
        idx_of_all_c = np.where(zs_first_frame == 6)[0]
        min_dist = 10
        bonded_c = -1  # Index of the C bonded to the hydrogen that gets abstracted
        for c in idx_of_all_c:
            dist = np.linalg.norm((xyz_first_frame[c]-xyz_first_frame[abstracted_h]))  # distance between a H and the CN carbon
            if dist < min_dist:
                min_dist = dist
                bonded_c = int(c)

        c_id[idx_frames] = bonded_c

    return h_id, c_id

def get_distances(xyz, h_id, c_id):
    ch_dist_alk = np.zeros(h_id.shape)
    ch_dist_cn = np.zeros(h_id.shape)

    # For each frame, calculate the distance between the h and the c in question
    for i in range(h_id.shape[0]):
        ch_dist_cn[i] = np.linalg.norm((xyz[i][h_id[i]]- xyz[i][-2]))
        ch_dist_alk[i] = np.linalg.norm((xyz[i][h_id[i]]- xyz[i][c_id[i]]))

    return ch_dist_alk, ch_dist_cn

def print_how_many_pst(h_id, traj_idx):
    """
    Counts how many trajectories are primary, secondary and tertiary abstractions.
    """
    # Where the hydrogens are primary, secondary or tertiary
    identity = [None, None, 1, 1, 1, 2, None, 2, None, 3, None, 1, 1, 1, 1, 1, 1, None, None]

    # Figuring out where each traj starts
    traj, idx_unique = np.unique(traj_idx, return_index=True)
    h_id_per_traj = h_id[idx_unique]

    primary = 0
    secondary = 0
    tertiary = 0

    for id in h_id_per_traj:
        if identity[id] == 1:
            primary += 1
        elif identity[id] == 2:
            secondary += 1
        elif identity[id] == 3:
            tertiary += 1

    print("There are %i, %i and %i primary, secondary and tertiary abstractions in %i trajectories" % (primary, secondary, tertiary, len(traj)) )

# Data dft
datasets = ['Methane', 'Ethane', 'Isobutane', 'Isopentane', '2Isohexane', '3Isohexane', 'Squalane']

for dataset in datasets:
    dataset_path_name = "../datasets/" + dataset.lower() + "_cn_dft.hdf5"
    dataset_path = glob.glob(dataset_path_name)[0]
    data = h5py.File(dataset_path, "r")

    traj_idx = np.array(data.get("traj_idx"))
    xyz = np.array(data.get("xyz"))
    zs = np.array(data.get("zs"))
    file_number = np.array(data.get("Filenumber"))

    # Sorting the trajectories
    idx_sorted = traj_idx.argsort()

    traj_idx = traj_idx[idx_sorted]
    file_number = file_number[idx_sorted]
    xyz = xyz[idx_sorted]
    zs = zs[idx_sorted]

    n_traj = np.unique(traj_idx)

    for item in n_traj:
        indices = np.where(traj_idx == item)

        idx_sorted = file_number[indices].argsort()
        traj_idx[indices] = traj_idx[indices][idx_sorted]
        file_number[indices] = file_number[indices][idx_sorted]
        xyz[indices] = xyz[indices][idx_sorted]
        zs[indices] = zs[indices][idx_sorted]

    # Finding out which H is abstracted
    h_id, c_id = get_h_c_abstracted(traj_idx, xyz, zs)
    ch_dist_alk_vr, ch_dist_cn_vr = get_distances(xyz, h_id, c_id)

    idx = list(range(traj_idx.shape[0]))
    shuffle(idx)

    fig, ax = plt.subplots(1, figsize=(8,6))
    ax.scatter(ch_dist_alk_vr[idx], ch_dist_cn_vr[idx], alpha=0.1)
    xlabel = "$C_{%s}$-$H$ distance (Å)"%dataset
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$C_{CN}$-$H$ distance (Å)")
    ax.set_xlim((0.5, 5.0))
    ax.set_ylim((0.5, 5.0))

    plt.tight_layout()
    # image_name = "scatter_" + dataset.lower() + ".png"
    # plt.savefig(image_path, dpi=200)
    plt.show()
