import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set()
sns.set_context("poster")
sns.set_style("white")

def update(frame, traj, line):
    line.set_xdata(x=traj[frame])

# Loading the predictons
traj_filepath = "../outputs/hc1sq_predict_001/sorted_predictions.npz"

prediction_data = np.load(traj_filepath)

# Get the number of the arrays
name = "arr_" + str(len(prediction_data) - 1)

# Get the arrays
true_values = prediction_data[name]

# Making the figures
fig, ax1 = plt.subplots(figsize=(12,6))

x = list(range(len(true_values)))
ax1.plot(x, true_values, label="DFT")
ax1.set_xlabel("Frame Number")
ax1.set_ylabel("Energy (kJ/mol)")
ax1.tick_params(axis='x')
ax1.tick_params(axis='y')

# cmap = mpl.cm.get_cmap('RdBu')
cmap = ListedColormap(sns.color_palette().as_hex())
line = ax1.axvline(x=0, color=cmap(1))

plt.tight_layout()

frames = range(0, len(x), 1)
ani = FuncAnimation(fig, update, frames, blit=False, interval=1, fargs=(x, line))

plt.rcParams['animation.ffmpeg_path'] = '/Users/walfits/anaconda3/envs/deffi/bin/ffmpeg'
writer = animation.FFMpegWriter(fps=60, bitrate=1800)
ani.save(filename='../images/squalane_traj.mp4', writer=writer, dpi=100)

# plt.show()
