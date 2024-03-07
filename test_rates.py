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
ref_beta = 1.0

ref_lamb_3d = ref_E * ref_nu / ((1.0 + ref_nu) * (1.0 - 2.0 * ref_nu))
ref_mu = ref_E / (2.0 * (1.0 + ref_nu))


def ref_phi_func(x, y, t):
    return np.cos(t) * np.cos(2.0 * np.pi * x) * np.cos(np.pi * y)


def ref_dt_phi_func(x, y, t):
    return -np.sin(t) * np.cos(2.0 * np.pi * x) * np.cos(np.pi * y)


def ref_theta_func(x, y, t):
    return np.sin(t) * np.cos(np.pi * x) * np.cos(2.0 * np.pi * y)


def ref_dt_theta_func(x, y, t):
    return np.cos(t) * np.cos(np.pi * x) * np.cos(2.0 * np.pi * y)


def ref_phi_source_func(x, y, t):
    phi, theta = ref_phi_func(x, y, t), ref_theta_func(x, y, t)
    val = ref_alpha * ref_dt_phi_func(x, y, t)
    val += ref_lambda_ * ref_epsilon * 5.0 * np.pi**2 * phi
    val += ref_lambda_ / ref_epsilon * Solver.W_prime_func(phi)
    val += ref_gamma * (theta - ref_theta_c) * Solver.p_func(theta)
    return val


def ref_theta_source_func(x, y, t):
    phi, theta = ref_phi_func(x, y, t), ref_theta_func(x, y, t)
    val = ref_delta * ref_dt_theta_func(x, y, t)
    val -= ref_gamma * Solver.p_func(phi) * ref_dt_phi_func(x, y, t)
    val += 5.0 * np.pi**2 * theta
    return val


def get_ref_phi_theta_discrete(sav_solver: Solver, t: float):
    phi_theta = np.zeros((sav_solver.dof_phi_theta_num))
    for i, j in product(range(sav_solver.N + 1), range(sav_solver.N + 1)):
        x, y = i * sav_solver.h, j * sav_solver.h
        nd_ind = j * (N + 1) + i
        phi_theta[nd_ind * 2] = ref_phi_func(x, y, t)
        phi_theta[nd_ind * 2 + 1] = ref_theta_func(x, y, t)
    return phi_theta


if __name__ == "__main__":
    N, steps = 32, 10
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
        ref_beta,
    )
    sav_solver.phi_source_func = ref_phi_source_func
    sav_solver.theta_source_func = ref_theta_source_func

    phi_theta_0 = get_ref_phi_theta_discrete(sav_solver, 0.0)
    q_0 = sav_solver.get_Q(phi_theta_0)

    phi_theta_ = np.zeros((sav_solver.dof_phi_theta_num))
    phi_theta = np.zeros((sav_solver.dof_phi_theta_num))
    q_, q = 0.0, 0.0

    phi_theta_[:] = phi_theta_0[:]
    q_ = sav_solver.get_Q(phi_theta_0)

    error_phi = np.zeros((sav_solver.steps, 2))
    error_theta = np.zeros((sav_solver.steps, 2))

    prg_bar = 0.0
    for i in range(1, sav_solver.steps + 1):
        phi_theta, q = sav_solver.get_next_phi_theta_q(phi_theta_, q_, i)
        phi_theta_ref = get_ref_phi_theta_discrete(sav_solver, i * sav_solver.tau)
        # Calculate errors
        diff = phi_theta - phi_theta_ref
        error_phi[i - 1, 0] = np.sqrt(
            np.dot(sav_solver.mass_mat.dot(diff[::2]), diff[::2])
        )
        # error_phi[i - 1, 0] /= np.sqrt(
        #     np.dot(sav_solver.mass_mat.dot(phi_theta_ref[::2]), phi_theta_ref[::2])
        # )
        error_phi[i - 1, 1] = np.sqrt(
            np.dot(sav_solver.stiff_mat.dot(diff[::2]), diff[::2])
        )
        # error_phi[i - 1, 1] /= np.sqrt(
        #     np.dot(sav_solver.stiff_mat.dot(phi_theta_ref[::2]), phi_theta_ref[::2])
        # )

        error_theta[i - 1, 0] = np.sqrt(
            np.dot(sav_solver.mass_mat.dot(diff[1::2]), diff[1::2])
        )
        # error_theta[i - 1, 0] /= np.sqrt(
        #     np.dot(sav_solver.mass_mat.dot(phi_theta_ref[1::2]), phi_theta_ref[1::2])
        # )
        error_theta[i - 1, 1] = np.sqrt(
            np.dot(sav_solver.stiff_mat.dot(diff[1::2]), diff[1::2])
        )
        # error_theta[i - 1, 1] /= np.sqrt(
        #     np.dot(sav_solver.stiff_mat.dot(phi_theta_ref[1::2]), phi_theta_ref[1::2])
        # )

        q_ = q
        phi_theta_ = phi_theta

        if (i / sav_solver.steps) > prg_bar:
            prg_bar += 0.1
            print(
                "{0:s}\t progress: {1:3.0f}%".format(
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    i / sav_solver.steps * 100.0,
                )
            )

    print(
        "Max error phi: in l_2={0:6e}, in h_1={1:6e}".format(
            np.max(error_phi[:, 0]), np.max(error_phi[:, 1])
        )
    )
    print(
        "Max error theta: in l_2={0:6e}, in h_1={1:6e}".format(
            np.max(error_theta[:, 0]), np.max(error_theta[:, 1])
        )
    )
