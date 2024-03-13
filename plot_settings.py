import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

plt.style.use("seaborn-v0_8-paper")

# Nature suggests that fontsizes should be between 5~7pt.
DEFAULT_FONT_SIZE = 7
SMALL_FONT_SIZE = 6
# plt.rc("text", usetex=True)
plt.rc("font", size=DEFAULT_FONT_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=DEFAULT_FONT_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=DEFAULT_FONT_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_FONT_SIZE)  # legend fontsize
plt.rc("figure", titlesize=DEFAULT_FONT_SIZE)  # fontsize of the figure title

A4_WIDTH = 6.5
# Nature suggests the width should be 180mm.
NATURE_WIDTH = 7.0866142

FIGS_ROOT_PATH = "resources"

# Some pdfs are too big! Maybe it is better to use pngs.
DPI = 1000


def plot_node_dat(dat: np.ndarray, ax, ran=None):
    # In our setting, dat[i, j] means the node at (x, y) = (j*h, i*h)
    xx = np.linspace(0.0, 1.0, dat.shape[1])
    yy = np.linspace(0.0, 1.0, dat.shape[0])
    if ran == None:
        posi = ax.pcolormesh(xx, yy, dat, shading="gouraud")
    else:
        posi = ax.pcolormesh(xx, yy, dat, shading="gouraud", vmin=ran[0], vmax=ran[1])
    ax.set_aspect("equal", "box")
    return posi


def append_colorbar(fig, ax, posi):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad="2%")
    cbar = fig.colorbar(posi, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=SMALL_FONT_SIZE, rotation=15)
    cax.xaxis.set_ticks_position("top")
    return cbar
