#Data sets

This folder contains the data sets used in the paper. 
For each of the hydrocarbons used (methane, ethane, isobutane, isopentane, 2-isohexane, 3-isohexane and squalane) there is a dataset with the energies at the PM6 and the DFT (Coulomb fitted unrestricted PB functional with the Def2-TZVP basis set) level.
For the dft datasets, the energies are in Hartrees, while for the PM6 they are in kcal/mol.

In snipped of code below shows how to extract data from the hdf5 file:

```
data_methane = h5py.File("methane_cn_dft.hdf5", "r")
xyz_methane = np.array(data_methane.get("xyz"))                     # np array of shape (n_samples, n_atoms, 3)
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)       # np array of shape (n_samples, n_atoms)
ene_methane = np.array(data_methane.get("ene"))                     # np array of shape (n_samples,)
traj_idx = np.array(data.get("traj_idx"))                           # np array of shape (n_samples,)
file_number = np.array(data.get("Filenumber"))                      # np array of shape (n_sampels,)
```
`traj_idx` is an index telling from which trajectory a particular sample came from.
`file_number` is an index telling which frame in a trajectory a particular sample is.