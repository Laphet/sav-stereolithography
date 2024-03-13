import plot_settings
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


data_moving = np.load(
    "{0:s}/{1:s}.npy".format(plot_settings.FIGS_ROOT_PATH, "moving-heat-source")
)
data_fixed = np.load(
    "{0:s}/{1:s}.npy".format(plot_settings.FIGS_ROOT_PATH, "fixed-heat-source")
)

N = 400
steps = 100
dof_phi_theta_num = 2 * (N + 1) ** 2
dof_ela_num = 2 * (N - 1) ** 2


# Retrieve data.
def get_phi_theta_ux_uy(data, i):
    phi = data[i, 0, :dof_phi_theta_num:2].reshape((N + 1, N + 1))
    theta = data[i, 0, 1:dof_phi_theta_num:2].reshape((N + 1, N + 1))
    ux_inner = data[i, 1, :dof_ela_num:2].reshape((N - 1, N - 1))
    uy_inner = data[i, 1, 1:dof_ela_num:2].reshape((N - 1, N - 1))
    ux = np.zeros((N + 1, N + 1))
    uy = np.zeros((N + 1, N + 1))
    ux[1:-1, 1:-1] = ux_inner
    uy[1:-1, 1:-1] = uy_inner
    return phi, theta, ux, uy


# data = data_fixed
# file_name_prefix = "fixed-heat-source"
data = data_moving
file_name_prefix = "moving-heat-source"

fig = plot_settings.plt.figure(
    figsize=(0.25 * plot_settings.A4_WIDTH, 0.35 * plot_settings.A4_WIDTH),
    layout="constrained",
)
for i in range(0, steps, 10):
    t = (i + 1) / steps
    phi, theta, ux, uy = get_phi_theta_ux_uy(data, i)

    # Phi
    ax = fig.add_subplot()
    posi = plot_settings.plot_node_dat(phi, ax, [-0.5, 0.5])
    cbar = plot_settings.append_colorbar(fig, ax, posi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
    ax.yaxis.set_label_coords(-0.1, 0.5)
    fig.suptitle(r"$\varphi$ at $t={:.2f}$".format(t))
    fig.savefig(
        "{0:s}/{1:s}-{2:s}-{3:03d}.png".format(
            plot_settings.FIGS_ROOT_PATH, file_name_prefix, "phi", i
        ),
        dpi=plot_settings.DPI,
    )
    fig.clear()

    # Theta
    ax = fig.add_subplot()
    posi = plot_settings.plot_node_dat(theta, ax)
    cbar = plot_settings.append_colorbar(fig, ax, posi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
    ax.yaxis.set_label_coords(-0.1, 0.5)
    fig.suptitle(r"$\theta$ at $t={:.2f}$".format(t))
    fig.savefig(
        "{0:s}/{1:s}-{2:s}-{3:03d}.png".format(
            plot_settings.FIGS_ROOT_PATH, file_name_prefix, "theta", i
        ),
        dpi=plot_settings.DPI,
    )
    fig.clear()

    # ux
    ax = fig.add_subplot()
    posi = plot_settings.plot_node_dat(ux, ax)
    cbar = plot_settings.append_colorbar(fig, ax, posi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
    ax.yaxis.set_label_coords(-0.1, 0.5)
    fig.suptitle(r"$u_x$ at $t={:.2f}$".format(t))
    fig.savefig(
        "{0:s}/{1:s}-{2:s}-{3:03d}.png".format(
            plot_settings.FIGS_ROOT_PATH, file_name_prefix, "ux", i
        ),
        dpi=plot_settings.DPI,
    )
    fig.clear()

    # uy
    ax = fig.add_subplot()
    posi = plot_settings.plot_node_dat(uy, ax)
    cbar = plot_settings.append_colorbar(fig, ax, posi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xticks([0.0, 1.0], ["0.0", "1.0"])
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_yticks([0.0, 1.0], ["0.0", "1.0"])
    ax.yaxis.set_label_coords(-0.1, 0.5)
    fig.suptitle(r"$u_y$ at $t={:.2f}$".format(t))
    fig.savefig(
        "{0:s}/{1:s}-{2:s}-{3:03d}.png".format(
            plot_settings.FIGS_ROOT_PATH, file_name_prefix, "uy", i
        ),
        dpi=plot_settings.DPI,
    )
    fig.clear()
