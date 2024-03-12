import plot_settings
import numpy as np

# Data from the test.
h_list_str = ["1/8", "1/16", "1/32", "1/64", "1/128"]
h_list_s_str = ["1/100", "1/200"]
tau_list_str = ["1/10", "1/20", "1/40", "1/80", "1/160"]
tau_list_s_str = ["1/100", "1/200"]

h_list = np.array([1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128])
h_list_s = np.array([1 / 100, 1 / 200])
tau_list = np.array([1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160])
tau_list_s = np.array([1 / 100, 1 / 200])

# Data format, (h_list) x (tau_list_s) x (phi, theta, u_x, u_y) x (l2, h1)
errors_N = np.array(
    [
        [
            [
                [4.888772e-02, 2.626673e-01],
                [1.720267e-02, 1.148457e-01],
                [2.705284e-02, 2.484043e-01],
                [1.698660e-02, 1.664432e-01],
            ],
            [
                [5.281721e-02, 2.694955e-01],
                [1.753448e-02, 1.144067e-01],
                [2.729726e-02, 2.499852e-01],
                [1.708007e-02, 1.670506e-01],
            ],
        ],
        [
            [
                [1.628225e-02, 9.889520e-02],
                [4.672487e-03, 3.034543e-02],
                [1.198443e-02, 1.174894e-01],
                [8.864237e-03, 9.919517e-02],
            ],
            [
                [1.632210e-02, 9.003624e-02],
                [4.756551e-03, 3.006447e-02],
                [1.194868e-02, 1.172927e-01],
                [8.826341e-03, 1.002151e-01],
            ],
        ],
        [
            [
                [6.936336e-03, 5.022380e-02],
                [1.251842e-03, 8.093231e-03],
                [6.047956e-03, 6.602851e-02],
                [3.741260e-03, 3.742820e-02],
            ],
            [
                [5.404537e-03, 3.496759e-02],
                [1.240083e-03, 7.775299e-03],
                [6.108369e-03, 6.692330e-02],
                [3.941973e-03, 3.894624e-02],
            ],
        ],
        [
            [
                [4.962902e-03, 3.899361e-02],
                [4.237122e-04, 2.811165e-03],
                [1.862953e-03, 2.221457e-02],
                [1.522962e-03, 1.626565e-02],
            ],
            [
                [2.894640e-03, 2.198409e-02],
                [3.537135e-04, 2.247435e-03],
                [2.216897e-03, 2.300872e-02],
                [1.670636e-03, 1.895440e-02],
            ],
        ],
        [
            [
                [4.574472e-03, 3.644829e-02],
                [2.729696e-04, 1.886719e-03],
                [1.057565e-03, 1.105453e-02],
                [7.028542e-04, 7.266266e-03],
            ],
            [
                [2.424956e-03, 1.920677e-02],
                [1.631113e-04, 1.098268e-03],
                [1.095920e-03, 1.119107e-02],
                [7.480604e-04, 7.499243e-03],
            ],
        ],
    ]
)

# Data format, (h_list_s) x (tau_list) x (phi, theta, u_x, u_y) x (l2, h1)
errors_steps = np.array(
    [
        [
            [
                [2.915411e-02, 2.378970e-01],
                [1.499161e-03, 1.054052e-02],
                [2.381746e-03, 2.880946e-02],
                [2.391191e-03, 2.742575e-02],
            ],
            [
                [1.811256e-02, 1.460394e-01],
                [9.696206e-04, 6.809596e-03],
                [1.463326e-03, 1.740908e-02],
                [1.284322e-03, 1.481681e-02],
            ],
            [
                [1.038472e-02, 8.316345e-02],
                [5.872307e-04, 4.104746e-03],
                [1.168457e-03, 1.413857e-02],
                [1.147758e-03, 1.212160e-02],
            ],
            [
                [5.676969e-03, 4.521902e-02],
                [3.488837e-04, 2.404635e-03],
                [1.128627e-03, 1.348336e-02],
                [9.411658e-04, 9.811471e-03],
            ],
            [
                [3.058628e-03, 2.416271e-02],
                [2.215250e-04, 1.478892e-03],
                [1.062862e-03, 1.319779e-02],
                [8.462341e-04, 9.313735e-03],
            ],
        ],
        [
            [
                [2.904228e-02, 2.370721e-01],
                [1.467710e-03, 1.031966e-02],
                [2.397038e-03, 2.915485e-02],
                [2.382869e-03, 2.765221e-02],
            ],
            [
                [1.799118e-02, 1.451409e-01],
                [9.415887e-04, 6.616916e-03],
                [1.475255e-03, 1.647841e-02],
                [1.247430e-03, 1.444093e-02],
            ],
            [
                [1.025410e-02, 8.221197e-02],
                [5.572133e-04, 3.909043e-03],
                [8.327439e-04, 9.275729e-03],
                [6.693140e-04, 7.761173e-03],
            ],
            [
                [5.535279e-03, 4.422238e-02],
                [3.108106e-04, 2.172408e-03],
                [5.274306e-04, 5.622096e-03],
                [4.882817e-04, 5.207026e-03],
            ],
            [
                [2.901183e-03, 2.311668e-02],
                [1.690624e-04, 1.172467e-03],
                [4.182777e-04, 5.120980e-03],
                [3.826997e-04, 3.949395e-03],
            ],
        ],
    ]
)

# Plot of phi.
fig = plot_settings.plt.figure(
    figsize=(plot_settings.A4_WIDTH, 0.5 * plot_settings.A4_WIDTH), layout="constrained"
)
ax = fig.add_subplot(1, 2, 1)
ax.plot(
    h_list,
    errors_N[:, 0, 0, 0],
    marker="D",
    label=r"$\|\cdot\|_0$, $\tau$=" + tau_list_s_str[0],
)
ax.plot(
    h_list,
    errors_N[:, 0, 0, 1],
    marker="d",
    label=r"$|\cdot|_1$, $\tau$=" + tau_list_s_str[0],
)
ax.plot(
    h_list,
    errors_N[:, 1, 0, 0],
    marker="H",
    label=r"$\|\cdot\|_0$, $\tau$=" + tau_list_s_str[1],
)
ax.plot(
    h_list,
    errors_N[:, 1, 0, 1],
    marker="h",
    label=r"$|\cdot|_1$, $\tau$=" + tau_list_s_str[1],
)
ax.plot(
    h_list[:3],
    h_list[:3] ** 2 * 5.0,
    linestyle="--",
    label=r"$O(h^2)$",
)
ax.plot(
    h_list[:3],
    h_list[:3] * 1.5,
    linestyle="--",
    label=r"$O(h)$",
)

ax.set_xlabel("$h$")
ax.set_xscale("log", base=2)
ax.set_xticks(h_list, h_list_str)
ax.set_ylabel("error")
ax.set_yscale("log")
ax.legend()
ax.set_title("(a)", fontweight="bold")

ax = fig.add_subplot(1, 2, 2)
ax.plot(
    tau_list,
    errors_steps[0, :, 0, 0],
    marker="D",
    label="$\|\cdot\|_0$, $h$=" + h_list_s_str[0],
)
ax.plot(
    tau_list,
    errors_steps[0, :, 0, 1],
    marker="d",
    label="$|\cdot|_1$, $h$=" + h_list_s_str[0],
)
ax.plot(
    tau_list,
    errors_steps[1, :, 0, 0],
    marker="H",
    label="$\|\cdot\|_0$, $h$=" + h_list_s_str[1],
)
ax.plot(
    tau_list,
    errors_steps[1, :, 0, 1],
    marker="h",
    label="$|\cdot|_1$, $h$=" + h_list_s_str[1],
)
ax.plot(
    tau_list[:3],
    tau_list[:3] ** 2 * 10.0,
    linestyle="--",
    label=r"$O(\tau^2)$",
)
ax.plot(
    tau_list[:3],
    tau_list[:3] * 1.5,
    linestyle="--",
    label=r"$O(\tau)$",
)

ax.set_xlabel(r"$\tau$")
ax.set_xscale("log", base=10)
ax.set_xticks(tau_list, tau_list_str)
ax.set_ylabel("error")
ax.set_yscale("log")
ax.legend()
ax.set_title("(b)", fontweight="bold")

fig.savefig(
    "{0:s}/{1:s}.pdf".format(plot_settings.FIGS_ROOT_PATH, "test-rates-phi"),
    bbox_inches="tight",
)

# Plot of theta.
fig = plot_settings.plt.figure(
    figsize=(plot_settings.A4_WIDTH, 0.5 * plot_settings.A4_WIDTH), layout="constrained"
)
ax = fig.add_subplot(1, 2, 1)
ax.plot(
    h_list,
    errors_N[:, 0, 1, 0],
    marker="D",
    label=r"$\|\cdot\|_0$, $\tau=" + tau_list_s_str[0] + "$",
)
ax.plot(
    h_list,
    errors_N[:, 0, 1, 1],
    marker="d",
    label=r"$|\cdot|_1$, $\tau=" + tau_list_s_str[0] + "$",
)
ax.plot(
    h_list,
    errors_N[:, 1, 1, 0],
    marker="H",
    label=r"$\|\cdot\|_0$, $\tau=" + tau_list_s_str[1] + "$",
)
ax.plot(
    h_list,
    errors_N[:, 1, 1, 1],
    marker="h",
    label=r"$|\cdot|_1$, $\tau=" + tau_list_s_str[1] + "$",
)
ax.plot(
    h_list[:3],
    h_list[:3] ** 2 * 0.5,
    linestyle="--",
    label=r"$O(h^2)$",
)
ax.plot(
    h_list[:3],
    h_list[:3] * 0.5,
    linestyle="--",
    label=r"$O(h)$",
)

ax.set_xlabel("$h$")
ax.set_xscale("log", base=2)
ax.set_xticks(h_list, h_list_str)
ax.set_ylabel("error")
ax.set_yscale("log")
ax.legend()
ax.set_title("(a)", fontweight="bold")

ax = fig.add_subplot(1, 2, 2)
ax.plot(
    tau_list,
    errors_steps[0, :, 1, 0],
    marker="D",
    label="$\|\cdot\|_0$, $h=" + h_list_s_str[0] + "$",
)
ax.plot(
    tau_list,
    errors_steps[0, :, 1, 1],
    marker="d",
    label="$|\cdot|_1$, $h=" + h_list_s_str[0] + "$",
)
ax.plot(
    tau_list,
    errors_steps[1, :, 1, 0],
    marker="H",
    label="$\|\cdot\|_0$, $h=" + h_list_s_str[1] + "$",
)
ax.plot(
    tau_list,
    errors_steps[1, :, 1, 1],
    marker="h",
    label="$|\cdot|_1$, $h=" + h_list_s_str[1] + "$",
)
ax.plot(
    tau_list[:3],
    tau_list[:3] ** 2 * 0.3,
    linestyle="--",
    label=r"$O(\tau^2)$",
)
ax.plot(
    tau_list[:3],
    tau_list[:3] * 0.1,
    linestyle="--",
    label=r"$O(\tau)$",
)

ax.set_xlabel(r"$\tau$")
ax.set_xscale("log", base=10)
ax.set_xticks(tau_list, tau_list_str)
ax.set_ylabel("error")
ax.set_yscale("log")
ax.legend()
ax.set_title("(b)", fontweight="bold")

fig.savefig(
    "{0:s}/{1:s}.pdf".format(plot_settings.FIGS_ROOT_PATH, "test-rates-theta"),
    bbox_inches="tight",
)

# Plot of u.
fig = plot_settings.plt.figure(
    figsize=(plot_settings.A4_WIDTH, 0.5 * plot_settings.A4_WIDTH), layout="constrained"
)
ax = fig.add_subplot(1, 2, 1)
ax.plot(
    h_list,
    np.sqrt(errors_N[:, 0, 2, 0] ** 2 + errors_N[:, 0, 3, 0] ** 2),
    marker="D",
    label=r"$\|\cdot\|_0$, $\tau=" + tau_list_s_str[0] + "$",
)
ax.plot(
    h_list,
    np.sqrt(errors_N[:, 0, 2, 1] ** 2 + errors_N[:, 0, 3, 1] ** 2),
    marker="d",
    label=r"$|\cdot|_1$, $\tau=" + tau_list_s_str[0] + "$",
)
ax.plot(
    h_list,
    np.sqrt(errors_N[:, 1, 2, 0] ** 2 + errors_N[:, 1, 3, 0] ** 2),
    marker="H",
    label=r"$\|\cdot\|_0$, $\tau=" + tau_list_s_str[1] + "$",
)
ax.plot(
    h_list,
    np.sqrt(errors_N[:, 1, 2, 1] ** 2 + errors_N[:, 1, 3, 1] ** 2),
    marker="h",
    label=r"$|\cdot|_1$, $\tau=" + tau_list_s_str[1] + "$",
)
ax.plot(
    h_list[:3],
    h_list[:3] ** 2 * 4.0,
    linestyle="--",
    label=r"$O(h^2)$",
)
ax.plot(
    h_list[:3],
    h_list[:3] * 2.0,
    linestyle="--",
    label=r"$O(h)$",
)

ax.set_xlabel("$h$")
ax.set_xscale("log", base=2)
ax.set_xticks(h_list, h_list_str)
ax.set_ylabel("error")
ax.set_yscale("log")
ax.legend()
ax.set_title("(a)", fontweight="bold")

ax = fig.add_subplot(1, 2, 2)
ax.plot(
    tau_list,
    np.sqrt(errors_steps[0, :, 2, 0] ** 2 + errors_steps[0, :, 3, 0] ** 2),
    marker="D",
    label="$\|\cdot\|_0$, $h=" + h_list_s_str[0] + "$",
)
ax.plot(
    tau_list,
    np.sqrt(errors_steps[0, :, 2, 1] ** 2 + errors_steps[0, :, 3, 1] ** 2),
    marker="d",
    label="$|\cdot|_1$, $h=" + h_list_s_str[0] + "$",
)
ax.plot(
    tau_list,
    np.sqrt(errors_steps[1, :, 2, 0] ** 2 + errors_steps[1, :, 3, 0] ** 2),
    marker="H",
    label="$\|\cdot\|_0$, $h=" + h_list_s_str[1] + "$",
)
ax.plot(
    tau_list,
    np.sqrt(errors_steps[1, :, 2, 1] ** 2 + errors_steps[1, :, 3, 1] ** 2),
    marker="h",
    label="$|\cdot|_1$, $h=" + h_list_s_str[1] + "$",
)
ax.plot(
    tau_list[:3],
    tau_list[:3] ** 2,
    linestyle="--",
    label=r"$O(\tau^2)$",
)
ax.plot(
    tau_list[:3],
    tau_list[:3] * 0.2,
    linestyle="--",
    label=r"$O(\tau)$",
)

ax.set_xlabel(r"$\tau$")
ax.set_xscale("log", base=10)
ax.set_xticks(tau_list, tau_list_str)
ax.set_ylabel("error")
ax.set_yscale("log")
ax.legend()
ax.set_title("(b)", fontweight="bold")

fig.savefig(
    "{0:s}/{1:s}.pdf".format(plot_settings.FIGS_ROOT_PATH, "test-rates-u"),
    bbox_inches="tight",
)
