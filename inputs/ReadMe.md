# Input scripts

This folder contains the scripts used to obtain the results presented in the paper.
Below is a short explanation of the various scripts.

- [make_scaler_001.py](./make_scaler_001.py): Fitting the Lasso model implemented in QML. The model obtained (`scaler.pickle`) can then be used to calculate the value to shift the energies of all the different hydrocarbons.
- [hc1sq_train_001.py](./hc1sq_train_001.py): Training an atomic neural network (ANN) on the training set containing only methane (Training set 1 in the paper).
- [hc2sq_train_001.py](./hc2sq_train_001.py): Training an ANN on the training set containing methane and ethane (Training set 2 in the paper).
- [hc3sq_train_001.py](./hc3sq_train_001.py): Training an ANN on the training set containing methane, ethane  and isobutane (Training set 3 in the paper).
- [hc4sq_train_001.py](./hc4sq_train_001.py): Training an ANN on the training set containing methane, ethane, isobutane and isopentane (Training set 4 in the paper).
- [hc5sq_train_001.py](./hc5sq_train_001.py): Training an ANN on the training set containing methane and isopentane (Training set 5 in the paper).
- [hc6sq_train_001.py](./hc6sq_train_001.py): Training an ANN on the training set containing methane, isopentane, 2-isohexane and 3-isohexane (Training set 6 in the paper).
- [hc1sq_predict_001.py](./hc1sq_train_001.py): Using the models trained by [hc1sq_train_001.py](./hc1sq_train_001.py), the energies of squalane are predicted.
- [hc2sq_predict_001.py](./hc2sq_train_001.py): Using the models trained by [hc2sq_train_001.py](./hc2sq_train_001.py), the energies of squalane are predicted.
- [hc3sq_predict_001.py](./hc3sq_train_001.py): Using the models trained by [hc3sq_train_001.py](./hc3sq_train_001.py), the energies of squalane are predicted.
- [hc4sq_predict_001.py](./hc4sq_train_001.py): Using the models trained by [hc4sq_train_001.py](./hc4sq_train_001.py), the energies of squalane are predicted.
- [hc5sq_predict_001.py](./hc5sq_train_001.py): Using the models trained by [hc5sq_train_001.py](./hc5sq_train_001.py), the energies of squalane are predicted.
- [hc6sq_predict_001.py](./hc6sq_train_001.py): Using the models trained by [hc6sq_train_001.py](./hc6sq_train_001.py), the energies of squalane are predicted.
- [timing_pred_001.py](./timing_pred_001.py): Script used to time the energy predictions for squalane with the NNs.
- [compare_pm6_001.py](./compare_pm6_001.py): Script used to compare the squalane energy predictions of the NN, DFT and PM6.

The calculations were run using this version of the [QML python package](https://github.com/qmlcode/qml):
Git commit reference: 5d13b78e483ba90f97b61cb5e4cf8e1abd251d9d