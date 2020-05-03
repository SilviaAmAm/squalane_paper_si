import numpy as np
import glob
import tensorflow as tf
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set()
sns.set_context("poster")
sns.set_style("white")
# sns.set_palette("colorblind")

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


    # Removing the offset
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None])
        Y = tf.placeholder(tf.float32, [None])
        
        c = tf.Variable(np.random.rand(1), name='c', dtype=tf.float32)
        model = tf.subtract(X, c)
        cost_function = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(model - Y, 2)))
        optimiser = tf.train.GradientDescentOptimizer(0.0001).minimize(cost_function)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init)
        for j in range(100):
            sess.run(optimiser, feed_dict={X: pred_values, Y: true_values})
            cost = sess.run(cost_function, feed_dict={X: pred_values, Y: true_values})
            if j%20 == 0:
                print("Step:", '%04d' % (j+1), "cost=", "{:.9f}".format(cost), "c=", sess.run(c))

        new_c = sess.run(c)[0]

    corrected_pred_values = pred_values - new_c

    # Calculating the R2 values
    r2 = r2_score(true_values, corrected_pred_values)
    print("R2 %s: %.2f" % (filepath.split("/")[-2], r2))

    # Making the figures
    fig = plt.figure(constrained_layout=True, figsize=(16,5))
    gs = GridSpec(1, 3, figure=fig)

    x = np.asarray(range(len(true_values)))/2
    ax_font = {'size': '19'}
    ax1 = fig.add_subplot(gs[:2])
    ax2 = fig.add_subplot(gs[2])
    ax1.plot(x, true_values, label="DFT")
    ax1.plot(x, pred_values, label="NN")
    ax1.set_xlabel("Time (fs)")
    ax1.set_ylabel("Energy (kJ/mol)")
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.legend()

    ax2.scatter(true_values, pred_values, s=10, c=sns.color_palette()[2])
    ax2.set_ylabel("NN energy (kJ/mol)")
    ax2.set_xlabel("DFT energy (kJ/mol)")
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.text(np.min(true_values),np.max(corrected_pred_values)-15, "R$^2$ = %.2f"%r2)
    # ax2.set_aspect(1)

    plt.savefig("../images/%isqual_wR2.png"%(i+1), dpi=200)
    # plt.show()
    # exit()
