import numpy as np
from sav_solver import Solver
from datetime import datetime


w_0 = 0.015
I_max = 4.0e4
t_check_pnts = [1.0 / 3, 2.0 / 3, 1.0]
coord_pnts = np.array(
    [[0.25, 0.5 + 1.0 / 3], [0.5, 0.5], [0.5, 0.5 - 1.0 / 3], [0.75, 0.5 + 1.0 / 3]]
)


# Moving center
def moving_heat_source(x, y, t):
    # t = (2.0 * t) % 1
    if t <= t_check_pnts[0]:
        delta_t = t / t_check_pnts[0]
        center = (1.0 - delta_t) * coord_pnts[0] + delta_t * coord_pnts[1]
        return I_max * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / w_0**2)
    elif t_check_pnts[0] < t <= t_check_pnts[1]:
        delta_t = (t - t_check_pnts[0]) / (t_check_pnts[1] - t_check_pnts[0])
        center = (1.0 - delta_t) * coord_pnts[2] + delta_t * coord_pnts[1]
        return I_max * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / w_0**2)
    elif t > t_check_pnts[1]:
        delta_t = (t - t_check_pnts[1]) / (1.0 - t_check_pnts[1])
        center = (1.0 - delta_t) * coord_pnts[3] + delta_t * coord_pnts[1]
        return I_max * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / w_0**2)


# Fixed center
def fixed_heat_source(x, y, t):
    # I_t = np.sin(np.pi * t) * I_max
    I_t = 1.0 * I_max
    return I_t * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / w_0**2)


ref_alpha = 0.5
ref_lambda_ = 1.0
# For h = 1/400
ref_epsilon = 0.5e-2
# For h = 1/200
# ref_epsilon = 1.0e-2
ref_gamma = 4.0e2
ref_theta_c = 1.0
ref_delta = 1.0e2

ref_kappa = 1.0e-6
ref_phi_gel = 0.5
ref_E = 1.0e4
ref_nu = 0.35
ref_zeta = 1.0e3
ref_beta = 0.5e3

N = 400
steps = 100

if __name__ == "__main__":
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
    # Use moving_heat_source or fixed_heat_source
    # sav_solver.theta_source_func = fixed_heat_source
    # file_name_prefix = "fixed-heat-source"
    sav_solver.theta_source_func = moving_heat_source
    file_name_prefix = "moving-heat-source"
    file_name_postfix = "r2"

    data_to_save = np.zeros(
        (steps + 1, 2, max(sav_solver.dof_phi_theta_num, sav_solver.dof_ela_num))
    )
    data_to_save[-1, 0, :14] = np.array(
        [
            sav_solver.alpha,
            sav_solver.lambda_,
            sav_solver.epsilon,
            sav_solver.gamma,
            sav_solver.theta_c,
            sav_solver.delta,
            sav_solver.kappa,
            sav_solver.phi_gel,
            sav_solver.E,
            sav_solver.nu,
            sav_solver.zeta,
            sav_solver.beta,
            w_0,
            I_max,
        ]
    )

    # Initial condition
    phi_theta_0 = np.zeros(sav_solver.dof_phi_theta_num)
    # Start from liquid resin
    phi_theta_0[::2] = -1.0

    phi_theta_ = np.zeros((sav_solver.dof_phi_theta_num))
    phi_theta_[:] = phi_theta_0[:]
    q_ = sav_solver.get_Q(phi_theta_)

    prg_bar = 0.0
    for i in range(1, sav_solver.steps + 1):
        # Give phi, theta, q
        phi_theta, q = sav_solver.get_next_phi_theta_q(phi_theta_, q_, i)
        sav_solver.cut_phi(phi_theta)
        u = sav_solver.get_next_u(phi_theta, phi_theta_0, i)

        # Save data
        data_to_save[i - 1, 0, : sav_solver.dof_phi_theta_num] = phi_theta[:]
        data_to_save[i - 1, 1, : sav_solver.dof_ela_num] = u[:]
        data_to_save[i - 1, 1, -1] = q

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

    np.save(
        "{0:s}/{1:s}-{2:s}".format("resources", file_name_prefix, file_name_postfix),
        data_to_save,
    )
