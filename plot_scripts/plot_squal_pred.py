import numpy as np
import glob
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set()
sns.set_context("poster")
sns.set_style("white")

def add_subplot_axes(ax, rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

# Loading the predictons
prediction_filepaths = glob.glob("../outputs/hc*sq_predict_001/")
prediction_filepaths.sort()

for i, filepath in enumerate(prediction_filepaths):
    filepath += "sorted_predictions.npz"

    prediction_data = np.load(filepath)

    # Get the number of the last 2 arrays
    name1 = "arr_" + str(len(prediction_data)-2)
    name2 = "arr_" + str(len(prediction_data) - 1)

    # Get the arrays
    pred_values = prediction_data[name1]
    true_values = prediction_data[name2]

    # Calculating the R2 values
    r2 = r2_score(true_values, pred_values)

    # Making the figures
    fig = plt.figure(constrained_layout=True, figsize=(16,5))
    gs = GridSpec(1, 3, figure=fig)

    x = list(range(len(true_values)))
    ax_font = {'size': '19'}
    ax1 = fig.add_subplot(gs[:2])
    ax2 = fig.add_subplot(gs[2])
    ax1.plot(x, true_values, label="DFT")
    ax1.plot(x, pred_values, label="NN")
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("Energy (kJ/mol)")
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.legend()

    ax2.scatter(true_values, pred_values, s=10, c=sns.color_palette()[2])
    ax2.set_ylabel("NN energy (kJ/mol)")
    ax2.set_xlabel("DFT energy (kJ/mol)")
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.text(np.min(true_values), np.max(pred_values) - 15, "R$^2$ = %.2f" % r2)
    # ax2.set_aspect(1)

    plt.savefig("../images/%isqual_wR2.png" % (i + 1), dpi=200)
    # plt.show()
    # exit()