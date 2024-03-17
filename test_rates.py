import numpy as np
from itertools import product
from sav_solver import Solver
from datetime import datetime

ref_alpha = 1.0
ref_lambda_ = 1.0
ref_epsilon = 1.0
ref_gamma = 1.0
ref_theta_c = 0.0
ref_delta = 1.2

ref_kappa = 0.01
ref_phi_gel = 0.5
ref_E = 1.0
ref_nu = 0.3
ref_zeta = 1.0
ref_beta = 0.5

ref_lamb_3d = ref_E * ref_nu / ((1.0 + ref_nu) * (1.0 - 2.0 * ref_nu))
ref_mu = ref_E / (2.0 * (1.0 + ref_nu))


def ref_phi_func(x, y, t):
    return np.cos(t) * np.cos(2.0 * np.pi * x) * np.cos(np.pi * y)


def ref_dt_phi_func(x, y, t):
    return -np.sin(t) * np.cos(2.0 * np.pi * x) * np.cos(np.pi * y)


def ref_dx_phi_func(x, y, t):
    return -np.cos(t) * 2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.cos(np.pi * y)


def ref_dy_phi_func(x, y, t):
    return -np.cos(t) * np.pi * np.cos(2.0 * np.pi * x) * np.sin(np.pi * y)


def ref_laplace_phi_func(x, y, t):
    return -np.cos(t) * 5.0 * np.pi**2 * np.cos(2.0 * np.pi * x) * np.cos(np.pi * y)


def ref_theta_func(x, y, t):
    return np.sin(t) * np.cos(np.pi * x) * np.cos(2.0 * np.pi * y)


def ref_dt_theta_func(x, y, t):
    return np.cos(t) * np.cos(np.pi * x) * np.cos(2.0 * np.pi * y)


def ref_dx_theta_func(x, y, t):
    return -np.sin(t) * np.pi * np.sin(np.pi * x) * np.cos(2.0 * np.pi * y)


def ref_dy_theta_func(x, y, t):
    return -np.sin(t) * 2.0 * np.pi * np.cos(np.pi * x) * np.sin(2.0 * np.pi * y)


def ref_laplace_theta_func(x, y, t):
    return -np.sin(t) * 5.0 * np.pi**2 * np.cos(np.pi * x) * np.cos(2.0 * np.pi * y)


def ref_phi_source_func(x, y, t):
    phi, theta = ref_phi_func(x, y, t), ref_theta_func(x, y, t)
    val = ref_alpha * ref_dt_phi_func(x, y, t)
    val -= ref_lambda_ * ref_epsilon * ref_laplace_phi_func(x, y, t)
    val += ref_lambda_ / ref_epsilon * Solver.W_prime_func(phi)
    val += ref_gamma * (theta - ref_theta_c) * Solver.p_func(theta)
    return val


def ref_theta_source_func(x, y, t):
    phi = ref_phi_func(x, y, t)
    val = ref_delta * ref_dt_theta_func(x, y, t)
    val -= ref_gamma * Solver.p_func(phi) * ref_dt_phi_func(x, y, t)
    val -= ref_laplace_theta_func(x, y, t)
    return val


def get_ref_phi_theta_discrete(sav_solver: Solver, t: float):
    phi_theta = np.zeros((sav_solver.dof_phi_theta_num))
    for j, i in product(range(sav_solver.N + 1), range(sav_solver.N + 1)):
        x, y = i * sav_solver.h, j * sav_solver.h
        nd_ind = j * (N + 1) + i
        phi_theta[nd_ind * 2] = ref_phi_func(x, y, t)
        phi_theta[nd_ind * 2 + 1] = ref_theta_func(x, y, t)
    return phi_theta


def ref_k_phi(phi):
    if phi <= ref_phi_gel:
        return 0.0
    else:
        return (phi - ref_phi_gel) / (1.0 - ref_phi_gel)


def ref_z_phi(phi):
    return ref_kappa * ref_k_phi(phi) + 1.0 - ref_kappa


def ref_d_z_phi(phi):
    if phi <= ref_phi_gel:
        return 0.0
    else:
        return ref_kappa / (1.0 - ref_phi_gel)


def ref_ux_func(x, y, t):
    return np.sin(t) * np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)


def ref_dx_ux_func(x, y, t):
    return np.sin(t) * np.pi * np.cos(np.pi * x) * np.sin(2.0 * np.pi * y)


def ref_dy_ux_func(x, y, t):
    return np.sin(t) * 2.0 * np.pi * np.sin(np.pi * x) * np.cos(2.0 * np.pi * y)


def ref_dxx_ux_func(x, y, t):
    return -np.sin(t) * np.pi**2 * np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)


def ref_dyy_ux_func(x, y, t):
    return -np.sin(t) * 4.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)


def ref_dxy_ux_func(x, y, t):
    return np.sin(t) * 2.0 * np.pi**2 * np.cos(np.pi * x) * np.cos(2.0 * np.pi * y)


def ref_uy_func(x, y, t):
    return np.cos(t) * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)


def ref_dx_uy_func(x, y, t):
    return np.cos(t) * 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.sin(np.pi * y)


def ref_dy_uy_func(x, y, t):
    return np.cos(t) * np.pi * np.sin(2.0 * np.pi * x) * np.cos(np.pi * y)


def ref_dxx_uy_func(x, y, t):
    return -np.cos(t) * 4.0 * np.pi**2 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)


def ref_dyy_uy_func(x, y, t):
    return -np.cos(t) * np.pi**2 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)


def ref_dxy_uy_func(x, y, t):
    return np.cos(t) * 2.0 * np.pi**2 * np.cos(2.0 * np.pi * x) * np.cos(np.pi * y)


# m(x) = (x + 1.0) * 0.5 * zeta
def ref_A1_func(x, y, t):
    val = ref_dx_ux_func(x, y, t)
    val -= (ref_phi_func(x, y, t) + 1.0) * 0.5 * ref_zeta
    val += ref_beta * (ref_theta_func(x, y, t) - ref_theta_func(x, y, 0.0))
    return val


def ref_dx_A1_func(x, y, t):
    val = ref_dxx_ux_func(x, y, t)
    val -= ref_dx_phi_func(x, y, t) * 0.5 * ref_zeta
    val += ref_beta * (ref_dx_theta_func(x, y, t) - ref_dx_theta_func(x, y, 0.0))
    return val


def ref_dy_A1_func(x, y, t):
    val = ref_dxy_ux_func(x, y, t)
    val -= ref_dy_phi_func(x, y, t) * 0.5 * ref_zeta
    val += ref_beta * (ref_dy_theta_func(x, y, t) - ref_dy_theta_func(x, y, 0.0))
    return val


# m(x) = (x + 1.0) * 0.5 * zeta
def ref_A2_func(x, y, t):
    val = ref_dy_uy_func(x, y, t)
    val -= (ref_phi_func(x, y, t) + 1.0) * 0.5 * ref_zeta
    val += ref_beta * (ref_theta_func(x, y, t) - ref_theta_func(x, y, 0.0))
    return val


def ref_dx_A2_func(x, y, t):
    val = ref_dxy_uy_func(x, y, t)
    val -= ref_dx_phi_func(x, y, t) * 0.5 * ref_zeta
    val += ref_beta * (ref_dx_theta_func(x, y, t) - ref_dx_theta_func(x, y, 0.0))
    return val


def ref_dy_A2_func(x, y, t):
    val = ref_dyy_uy_func(x, y, t)
    val -= ref_dy_phi_func(x, y, t) * 0.5 * ref_zeta
    val += ref_beta * (ref_dy_theta_func(x, y, t) - ref_dy_theta_func(x, y, 0.0))
    return val


def ref_A3_func(x, y, t):
    return (ref_dx_uy_func(x, y, t) + ref_dy_ux_func(x, y, t)) * 0.5


def ref_dx_A3_func(x, y, t):
    return (ref_dxx_uy_func(x, y, t) + ref_dxy_ux_func(x, y, t)) * 0.5


def ref_dy_A3_func(x, y, t):
    return (ref_dxy_uy_func(x, y, t) + ref_dyy_ux_func(x, y, t)) * 0.5


def ref_u_source_func(x, y, t):
    phi = ref_phi_func(x, y, t)
    z_phi = ref_z_phi(phi)
    d_z_phi = ref_d_z_phi(phi)
    dx_z_phi, dy_z_phi = (
        d_z_phi * ref_dx_phi_func(x, y, t),
        d_z_phi * ref_dy_phi_func(x, y, t),
    )

    A1, A2, A3 = ref_A1_func(x, y, t), ref_A2_func(x, y, t), ref_A3_func(x, y, t)
    dx_z_A1 = dx_z_phi * A1 + z_phi * ref_dx_A1_func(x, y, t)
    dx_z_A2 = dx_z_phi * A2 + z_phi * ref_dx_A2_func(x, y, t)
    dy_z_A3 = dy_z_phi * A3 + z_phi * ref_dy_A3_func(x, y, t)
    dy_z_A1 = dy_z_phi * A1 + z_phi * ref_dy_A1_func(x, y, t)
    dy_z_A2 = dy_z_phi * A2 + z_phi * ref_dy_A2_func(x, y, t)
    dx_z_A3 = dx_z_phi * A3 + z_phi * ref_dx_A3_func(x, y, t)
    return (
        -(ref_lamb_3d + 2.0 * ref_mu) * dx_z_A1
        - ref_lamb_3d * dx_z_A2
        - 2.0 * ref_mu * dy_z_A3,
        -(ref_lamb_3d + 2.0 * ref_mu) * dy_z_A2
        - ref_lamb_3d * dy_z_A1
        - 2.0 * ref_mu * dx_z_A3,
    )


def get_ref_u_discrete(sav_solver: Solver, t: float):
    u = np.zeros((sav_solver.dof_ela_num))
    for j, i in product(range(sav_solver.N - 1), range(sav_solver.N - 1)):
        x, y = (i + 1) * sav_solver.h, (j + 1) * sav_solver.h
        nd_ind = j * (N - 1) + i
        u[nd_ind * 2] = ref_ux_func(x, y, t)
        u[nd_ind * 2 + 1] = ref_uy_func(x, y, t)
    return u


def get_u_ext(u: np.ndarray, N):
    # No data is allocated, just reshape the array.
    u_sq = u.reshape((N - 1, N - 1))
    # Allocate the extended array.
    u_ext = np.zeros((N + 1, N + 1))
    # This is a deep copy, entries are copied.
    u_ext[1:N, 1:N] = u_sq
    return u_ext.reshape((-1,))


if __name__ == "__main__":
    N_list = [8, 16, 32, 64, 128]
    N_list_s = [100, 200]
    steps_list = [10, 20, 40, 80, 160]
    steps_list_s = [100, 200]

    # N, steps = 32, 10
    # Get command line arguments
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "N", type=int, help="Number of elements in one direction", default=32
    # )
    # parser.add_argument("steps", type=int, help="Number of time steps", default=10)
    # args = parser.parse_args()
    # N, steps = args.N, args.steps

    for N, steps in product(N_list, steps_list):
        sav_solver = Solver(
            N,
            steps,
            ref_alpha,
            ref_lambda_,
            ref_epsilon,
            ref_gamma,
            ref_theta_c,
            ref_delta,
            ref_kappa,
            ref_phi_gel,
            ref_E,
            ref_nu,
            ref_zeta,
            ref_beta,
        )
        sav_solver.phi_source_func = ref_phi_source_func
        sav_solver.theta_source_func = ref_theta_source_func
        sav_solver.u_source_func = ref_u_source_func

        phi_theta_0 = get_ref_phi_theta_discrete(sav_solver, 0.0)
        q_0 = sav_solver.get_Q(phi_theta_0)

        phi_theta_ = np.zeros((sav_solver.dof_phi_theta_num))
        phi_theta = np.zeros((sav_solver.dof_phi_theta_num))
        q_, q = 0.0, 0.0

        phi_theta_[:] = phi_theta_0[:]
        q_ = sav_solver.get_Q(phi_theta_0)

        error_phi = np.zeros((sav_solver.steps, 2))
        error_theta = np.zeros((sav_solver.steps, 2))
        error_ux = np.zeros((sav_solver.steps, 2))
        error_uy = np.zeros((sav_solver.steps, 2))

        prg_bar = 0.0
        for i in range(1, sav_solver.steps + 1):
            # Give phi, theta, q
            phi_theta, q = sav_solver.get_next_phi_theta_q(phi_theta_, q_, i)
            # Calculate reference phi and theta
            phi_theta_ref = get_ref_phi_theta_discrete(sav_solver, i * sav_solver.tau)
            # Calculate phi and theta errors
            diff_phi_theta = phi_theta - phi_theta_ref
            error_phi[i - 1, 0] = np.sqrt(
                np.dot(
                    sav_solver.mass_mat.dot(diff_phi_theta[::2]), diff_phi_theta[::2]
                )
            )
            error_phi[i - 1, 1] = np.sqrt(
                np.dot(
                    sav_solver.stiff_mat.dot(diff_phi_theta[::2]), diff_phi_theta[::2]
                )
            )
            error_theta[i - 1, 0] = np.sqrt(
                np.dot(
                    sav_solver.mass_mat.dot(diff_phi_theta[1::2]), diff_phi_theta[1::2]
                )
            )
            error_theta[i - 1, 1] = np.sqrt(
                np.dot(
                    sav_solver.stiff_mat.dot(diff_phi_theta[1::2]), diff_phi_theta[1::2]
                )
            )

            # Get u
            u = sav_solver.get_next_u(phi_theta, phi_theta_0, i)
            # Calculate reference u
            u_ref = get_ref_u_discrete(sav_solver, i * sav_solver.tau)
            # Calculate u errors
            diff_u = u - u_ref
            diff_ux_ext = get_u_ext(diff_u[::2], sav_solver.N)
            diff_uy_ext = get_u_ext(diff_u[1::2], sav_solver.N)
            error_ux[i - 1, 0] = np.sqrt(
                np.dot(sav_solver.mass_mat.dot(diff_ux_ext), diff_ux_ext)
            )
            error_ux[i - 1, 1] = np.sqrt(
                np.dot(sav_solver.stiff_mat.dot(diff_ux_ext), diff_ux_ext)
            )
            error_uy[i - 1, 0] = np.sqrt(
                np.dot(sav_solver.mass_mat.dot(diff_uy_ext), diff_uy_ext)
            )
            error_uy[i - 1, 1] = np.sqrt(
                np.dot(sav_solver.stiff_mat.dot(diff_uy_ext), diff_uy_ext)
            )

            # Show progress
            if (i / sav_solver.steps) > prg_bar:
                prg_bar += 0.1
                print(
                    "{0:s}\tprogress: {1:3.0f}%".format(
                        datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        i / sav_solver.steps * 100.0,
                    )
                )

            # Update variables
            q_ = q
            phi_theta_ = phi_theta

        print(
            "Max error phi:  \tin l_2={0:6e}, in h_1={1:6e}".format(
                np.max(error_phi[:, 0]), np.max(error_phi[:, 1])
            )
        )
        print(
            "Max error theta:\tin l_2={0:6e}, in h_1={1:6e}".format(
                np.max(error_theta[:, 0]), np.max(error_theta[:, 1])
            )
        )
        print(
            "Max error ux:   \tin l_2={0:6e}, in h_1={1:6e}".format(
                np.max(error_ux[:, 0]), np.max(error_ux[:, 1])
            )
        )
        print(
            "Max error uy:   \tin l_2={0:6e}, in h_1={1:6e}".format(
                np.max(error_uy[:, 0]), np.max(error_uy[:, 1])
            )
        )
