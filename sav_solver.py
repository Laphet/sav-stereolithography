from itertools import product
import numpy as np
from scipy.sparse import csc_matrix

# The default sparse solver in scipy is superLU, with optional interfaces to UMFPACK.
# All those solvers are not parallelized.
# from scipy.sparse.linalg import spsolve
# Try to use the parallelized solver in the mkl library (Pardiso).
import pypardiso

spsolve = pypardiso.spsolve


QUAD_ORDER = 2
QUAD_CORD, QUAD_WGHT = np.polynomial.legendre.leggauss(QUAD_ORDER)
N_V = 4
DIM = 2
QUAD_PNTS = QUAD_ORDER**DIM
STRAIN_DOF = 3


def get_basis_val(loc_ind: int, x: float, y: float):
    val = -1.0
    if loc_ind == 0:
        val = 0.25 * (1.0 - x) * (1.0 - y)
    elif loc_ind == 1:
        val = 0.25 * (1.0 + x) * (1.0 - y)
    elif loc_ind == 2:
        val = 0.25 * (1.0 - x) * (1.0 + y)
    elif loc_ind == 3:
        val = 0.25 * (1.0 + x) * (1.0 + y)
    else:
        raise ValueError("Invalid option, loc_ind={:.d}".format(loc_ind))
    return val


def get_grad_basis_val(loc_ind: int, x: float, y: float):
    grad_val_x, grad_val_y = -1.0, -1.0
    if loc_ind == 0:
        grad_val_x, grad_val_y = -0.25 * (1.0 - y), -0.25 * (1.0 - x)
    elif loc_ind == 1:
        grad_val_x, grad_val_y = 0.25 * (1.0 - y), -0.25 * (1.0 + x)
    elif loc_ind == 2:
        grad_val_x, grad_val_y = -0.25 * (1.0 + y), 0.25 * (1.0 - x)
    elif loc_ind == 3:
        grad_val_x, grad_val_y = 0.25 * (1.0 + y), 0.25 * (1.0 + x)
    else:
        raise ValueError("Invalid option, loc_ind={:.d}".format(loc_ind))
    return grad_val_x, grad_val_y


basis_val_at_quad_pnt = np.zeros((N_V, QUAD_PNTS))
grad_basis_val_at_quad_pnt = np.zeros((N_V, 2, QUAD_PNTS))
quad_wghts = np.zeros((QUAD_PNTS,))

for loc_nd_ind, quad_pnt_ind_x, quad_pnt_ind_y in product(
    range(N_V), range(QUAD_ORDER), range(QUAD_ORDER)
):
    quad_pnt_ind = quad_pnt_ind_y * QUAD_ORDER + quad_pnt_ind_x
    x, y = QUAD_CORD[quad_pnt_ind_x], QUAD_CORD[quad_pnt_ind_y]
    basis_val_at_quad_pnt[loc_nd_ind, quad_pnt_ind] = get_basis_val(loc_nd_ind, x, y)
    grad_basis_val_at_quad_pnt[loc_nd_ind, :, quad_pnt_ind] = get_grad_basis_val(
        loc_nd_ind, x, y
    )
    quad_wghts[quad_pnt_ind] = QUAD_WGHT[quad_pnt_ind_x] * QUAD_WGHT[quad_pnt_ind_y]


# \varepsilon(u) [0:8, 0:3, :]
strain_val_at_quad_pnt = np.zeros((N_V * 2, STRAIN_DOF, QUAD_PNTS))
# \div u [0:8, :]
div_val_at_quad_pnt = np.zeros((N_V * 2, QUAD_PNTS))
for loc_nd_ind in range(N_V):
    # (u1, u2) = (phi, 0)
    strain_val_at_quad_pnt[loc_nd_ind * 2, 0, :] = grad_basis_val_at_quad_pnt[
        loc_nd_ind, 0, :
    ]
    strain_val_at_quad_pnt[loc_nd_ind * 2, 2, :] = (
        0.5 * grad_basis_val_at_quad_pnt[loc_nd_ind, 1, :]
    )
    div_val_at_quad_pnt[loc_nd_ind * 2, :] = grad_basis_val_at_quad_pnt[
        loc_nd_ind, 0, :
    ]
    # (u1, u2) = (0, phi)
    strain_val_at_quad_pnt[loc_nd_ind * 2 + 1, 1, :] = grad_basis_val_at_quad_pnt[
        loc_nd_ind, 1, :
    ]
    strain_val_at_quad_pnt[loc_nd_ind * 2 + 1, 2, :] = (
        0.5 * grad_basis_val_at_quad_pnt[loc_nd_ind, 0, :]
    )
    div_val_at_quad_pnt[loc_nd_ind * 2 + 1, :] = grad_basis_val_at_quad_pnt[
        loc_nd_ind, 1, :
    ]


# Element integral \int \grad(p) \cdot \grad(q)
elem_Laplace_stiff_mat = np.zeros((N_V, N_V))
# Element integral \int p q
elem_bilinear_mass_mat = np.zeros((N_V, N_V))
for loc_nd_i, loc_nd_j in product(range(N_V), range(N_V)):
    elem_Laplace_stiff_mat[loc_nd_i, loc_nd_j] = np.dot(
        grad_basis_val_at_quad_pnt[loc_nd_i, 0, :]
        * grad_basis_val_at_quad_pnt[loc_nd_j, 0, :]
        + grad_basis_val_at_quad_pnt[loc_nd_i, 1, :]
        * grad_basis_val_at_quad_pnt[loc_nd_j, 1, :],
        quad_wghts,
    )
    elem_bilinear_mass_mat[loc_nd_i, loc_nd_j] = np.dot(
        basis_val_at_quad_pnt[loc_nd_i, :] * basis_val_at_quad_pnt[loc_nd_j, :],
        quad_wghts,
    )

# Val \varepsilon(u) : \varepsilon(v) at quad points
ela_mu_stiff_at_quad_pnt = np.zeros((N_V * 2, N_V * 2, QUAD_PNTS))
# Val \div(u) \div(v) at quad points
ela_lambda_stiff_at_quad_pnt = np.zeros((N_V * 2, N_V * 2, QUAD_PNTS))
for loc_dof_i, loc_dof_j in product(range(N_V * 2), range(N_V * 2)):
    ela_mu_stiff_at_quad_pnt[loc_dof_i, loc_dof_j, :] = (
        strain_val_at_quad_pnt[loc_dof_i, 0, :]
        * strain_val_at_quad_pnt[loc_dof_j, 0, :]
        + strain_val_at_quad_pnt[loc_dof_i, 1, :]
        * strain_val_at_quad_pnt[loc_dof_j, 1, :]
        + 2.0
        * strain_val_at_quad_pnt[loc_dof_i, 2, :]
        * strain_val_at_quad_pnt[loc_dof_j, 2, :]
    )
    ela_lambda_stiff_at_quad_pnt[loc_dof_i, loc_dof_j, :] = (
        div_val_at_quad_pnt[loc_dof_i, :] * div_val_at_quad_pnt[loc_dof_j, :]
    )
# Element integral \int \varepsilon(u) : \varepsilon(v)
elem_ela_mu_stiff_mat = ela_mu_stiff_at_quad_pnt @ quad_wghts
# Element integral \int \div(u)  \div(v)
elem_ela_lambda_stiff_mat = ela_lambda_stiff_at_quad_pnt @ quad_wghts


def zero_source(x, y, t):
    return 0.0


def zero_source_2d(x, y, t):
    return 0.0, 0.0


class Solver:
    @staticmethod
    def W_func(x):
        return 0.25 * (x**2 - 1) ** 2

    @staticmethod
    def W_prime_func(x):
        return x**3 - x

    @staticmethod
    def P_func(x):
        return 0.5 * (1.0 - x)

    # p is the derivative of P
    @staticmethod
    def p_func(x):
        return -0.5

    def k(self, phi):
        if phi <= self.phi_gel:
            return 0.0
        else:
            return (phi - self.phi_gel) / (1.0 - self.phi_gel)

    # m = zeta * (1 - P)
    def m(self, x):
        return (x + 1.0) * 0.5 * self.zeta

    def __init__(
        self,
        N: int,
        steps: int,
        alpha: float = 1.0,
        lambda_: float = 1.0,
        epsilon: float = 1.0,
        gamma: float = 1.0,
        theta_c: float = 0.0,
        delta: float = 1.2,
        kappa: float = 0.01,
        phi_gel: float = 0.5,
        E: float = 1.0,
        nu: float = 0.3,
        zeta: float = 1.0,
        beta: float = 1.0,
    ):
        self.N = N
        self.steps = steps
        # In phase-temperature field
        self.alpha = alpha
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.gamma = gamma
        self.theta_c = theta_c
        self.delta = delta
        # In elasticity
        self.kappa = kappa
        self.phi_gel = phi_gel
        self.E = E
        self.nu = nu
        self.zeta = zeta
        self.beta = beta
        self.lamb_3d = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))

        # the function k contains "if"
        self.k_vec = np.vectorize(self.k)

        self.h = 1.0 / N
        self.tau = 1.0 / steps
        self.dof_phi_theta_num = 2 * (N + 1) ** 2
        self.dof_ela_num = 2 * (N - 1) ** 2

        max_data_len = self.N * self.N * N_V**2
        II = -np.ones((max_data_len,), dtype=np.int32)
        JJ = -np.ones((max_data_len,), dtype=np.int32)
        VV_stiff = np.zeros((max_data_len,))
        VV_mass = np.zeros((max_data_len,))
        marker = 0
        for elem_ind_x, elem_ind_y in product(range(self.N), range(self.N)):
            for loc_nd_row, loc_nd_col in product(range(N_V), range(N_V)):
                loc_nd_row_x, loc_nd_row_y = divmod(loc_nd_row, 2)
                loc_nd_col_x, loc_nd_col_y = divmod(loc_nd_col, 2)
                dof_ind_row = (
                    (elem_ind_y + loc_nd_row_y) * (self.N + 1)
                    + elem_ind_x
                    + loc_nd_row_x
                )
                dof_ind_col = (
                    (elem_ind_y + loc_nd_col_y) * (self.N + 1)
                    + elem_ind_x
                    + loc_nd_col_x
                )
                II[marker] = dof_ind_row
                JJ[marker] = dof_ind_col
                VV_stiff[marker] = elem_Laplace_stiff_mat[loc_nd_row, loc_nd_col]
                VV_mass[marker] = (
                    0.25 * self.h**2 * elem_bilinear_mass_mat[loc_nd_row, loc_nd_col]
                )
                marker += 1
        self.stiff_mat = csc_matrix(
            (VV_stiff[:marker], (II[:marker], JJ[:marker])),
            shape=((self.N + 1) ** 2, (self.N + 1) ** 2),
        )
        self.mass_mat = csc_matrix(
            (VV_mass[:marker], (II[:marker], JJ[:marker])),
            shape=((self.N + 1) ** 2, (self.N + 1) ** 2),
        )

        # phi_source_func, theta_source_func
        # To construct the exact solution
        self.phi_source_func = zero_source
        self.theta_source_func = zero_source
        self.u_source_func = zero_source_2d

        # Print the parameters
        print("*" * 80)
        print("This is a sav solver for a stereolithography model.")
        print(
            "Use N={:d}, steps={:d} => h={:.6e}, tau={:.6e}.".format(
                self.N, self.steps, self.h, self.tau
            )
        )
        print(
            (
                "In phase-temperature: alpha={:.6e}, lambda={:.6e}, epsilon={:.6e},\n"
                + "                      gamma={:.6e}, theta_c={:.6e}, delta={:.6e}."
            ).format(
                self.alpha,
                self.lambda_,
                self.epsilon,
                self.gamma,
                self.theta_c,
                self.delta,
            )
        )
        print(
            (
                "In elasticity:        kappa={:.6e}, phi_gel={:.6e}, E={:.6e},\n"
                + "                      nu={:.6e}, zeta={:.6e}, beta={:.6e}."
            ).format(self.kappa, self.phi_gel, self.E, self.nu, self.zeta, self.beta)
        )
        print("*" * 80)

    # A quick way to get Q
    def get_Q(self, phi_theta: np.ndarray):
        phi = phi_theta[::2]
        phi = phi.reshape((self.N + 1, self.N + 1))
        W_phi = self.W_func(phi)
        W_phi[0, :] *= 0.5
        W_phi[-1, :] *= 0.5
        W_phi[:, 0] *= 0.5
        W_phi[:, -1] *= 0.5
        return np.sqrt(1.0 / self.epsilon * np.sum(self.h**2 * W_phi) + 1.0)

    def get_next_phi_theta_q(self, phi_theta_: np.ndarray, q_: float, n: int):
        Q_ = self.get_Q(phi_theta_)

        max_data_len = self.N**2 * 64
        II = -np.ones(max_data_len, dtype=np.int32)
        JJ = -np.ones(max_data_len, dtype=np.int32)
        VV = np.zeros(max_data_len)

        rhs = np.zeros((self.dof_phi_theta_num,))
        W_prime_phi_over_eps_Q = np.zeros((self.dof_phi_theta_num,))

        marker = 0
        for elem_ind_x, elem_ind_y in product(range(self.N), range(self.N)):
            loc_nds = [
                elem_ind_y * (self.N + 1) + elem_ind_x,
                elem_ind_y * (self.N + 1) + elem_ind_x + 1,
                (elem_ind_y + 1) * (self.N + 1) + elem_ind_x,
                (elem_ind_y + 1) * (self.N + 1) + elem_ind_x + 1,
            ]
            loc_phi_ = np.array([phi_theta_[2 * loc_nd] for loc_nd in loc_nds])
            loc_theta_ = np.array([phi_theta_[2 * loc_nd + 1] for loc_nd in loc_nds])
            W_prime_phi_at_quad_pnt = self.W_prime_func(
                loc_phi_ @ basis_val_at_quad_pnt
            )
            loc_phi_source = np.array(
                [
                    self.phi_source_func(
                        elem_ind_x * self.h, elem_ind_y * self.h, n * self.tau
                    ),
                    self.phi_source_func(
                        (elem_ind_x + 1) * self.h, elem_ind_y * self.h, n * self.tau
                    ),
                    self.phi_source_func(
                        elem_ind_x * self.h, (elem_ind_y + 1) * self.h, n * self.tau
                    ),
                    self.phi_source_func(
                        (elem_ind_x + 1) * self.h,
                        (elem_ind_y + 1) * self.h,
                        n * self.tau,
                    ),
                ]
            )
            loc_theta_source = np.array(
                [
                    self.theta_source_func(
                        elem_ind_x * self.h, elem_ind_y * self.h, n * self.tau
                    ),
                    self.theta_source_func(
                        (elem_ind_x + 1) * self.h, elem_ind_y * self.h, n * self.tau
                    ),
                    self.theta_source_func(
                        elem_ind_x * self.h, (elem_ind_y + 1) * self.h, n * self.tau
                    ),
                    self.theta_source_func(
                        (elem_ind_x + 1) * self.h,
                        (elem_ind_y + 1) * self.h,
                        n * self.tau,
                    ),
                ]
            )
            p_phi_at_quad_pnt = self.p_func(loc_phi_ @ basis_val_at_quad_pnt)
            for loc_nd_row in range(N_V):
                for loc_nd_col in range(N_V):
                    # A_{phi, phi}
                    II[marker] = loc_nds[loc_nd_row] * 2
                    JJ[marker] = loc_nds[loc_nd_col] * 2
                    VV[marker] += (
                        self.alpha
                        / self.tau
                        * elem_bilinear_mass_mat[loc_nd_row, loc_nd_col]
                        * 0.25
                        * self.h**2
                    )
                    VV[marker] += (
                        self.lambda_
                        * self.epsilon
                        * elem_Laplace_stiff_mat[loc_nd_row, loc_nd_col]
                    )
                    marker += 1
                    # A_{phi, theta}
                    II[marker] = loc_nds[loc_nd_row] * 2
                    JJ[marker] = loc_nds[loc_nd_col] * 2 + 1
                    VV[marker] += (
                        self.gamma
                        * (
                            np.dot(
                                p_phi_at_quad_pnt
                                * basis_val_at_quad_pnt[loc_nd_row, :]
                                * basis_val_at_quad_pnt[loc_nd_col, :],
                                quad_wghts,
                            )
                        )
                        * 0.25
                        * self.h**2
                    )
                    marker += 1
                    # A_{theta, phi}
                    II[marker] = loc_nds[loc_nd_row] * 2 + 1
                    JJ[marker] = loc_nds[loc_nd_col] * 2
                    VV[marker] -= (
                        self.gamma
                        / self.tau
                        * (
                            np.dot(
                                p_phi_at_quad_pnt
                                * basis_val_at_quad_pnt[loc_nd_row, :]
                                * basis_val_at_quad_pnt[loc_nd_col, :],
                                quad_wghts,
                            )
                        )
                        * 0.25
                        * self.h**2
                    )
                    marker += 1
                    # A_{theta, theta}
                    II[marker] = loc_nds[loc_nd_row] * 2 + 1
                    JJ[marker] = loc_nds[loc_nd_col] * 2 + 1
                    VV[marker] += (
                        self.delta
                        / self.tau
                        * elem_bilinear_mass_mat[loc_nd_row, loc_nd_col]
                        * 0.25
                        * self.h**2
                    )
                    VV[marker] += elem_Laplace_stiff_mat[loc_nd_row, loc_nd_col]
                    marker += 1
                # Get loc rhs
                # rhs_{phi}
                rhs[loc_nds[loc_nd_row] * 2] += (
                    self.alpha
                    / self.tau
                    * np.dot(elem_bilinear_mass_mat[loc_nd_row, :], loc_phi_)
                    * 0.25
                    * self.h**2
                )
                rhs[loc_nds[loc_nd_row] * 2] += (
                    self.gamma
                    * self.theta_c
                    * np.dot(
                        p_phi_at_quad_pnt * basis_val_at_quad_pnt[loc_nd_row, :],
                        quad_wghts,
                    )
                    * 0.25
                    * self.h**2
                )
                rhs[loc_nds[loc_nd_row] * 2] += (
                    np.dot(elem_bilinear_mass_mat[loc_nd_row, :], loc_phi_source)
                    * 0.25
                    * self.h**2
                )
                # rhs_{theta}
                rhs[loc_nds[loc_nd_row] * 2 + 1] += (
                    self.delta
                    / self.tau
                    * np.dot(elem_bilinear_mass_mat[loc_nd_row, :], loc_theta_)
                    * 0.25
                    * self.h**2
                )
                rhs[loc_nds[loc_nd_row] * 2 + 1] -= (
                    self.gamma
                    / self.tau
                    * np.dot(
                        (loc_phi_ @ basis_val_at_quad_pnt)
                        * p_phi_at_quad_pnt
                        * basis_val_at_quad_pnt[loc_nd_row, :],
                        quad_wghts,
                    )
                    * 0.25
                    * self.h**2
                )
                rhs[loc_nds[loc_nd_row] * 2 + 1] += (
                    np.dot(elem_bilinear_mass_mat[loc_nd_row, :], loc_theta_source)
                    * 0.25
                    * self.h**2
                )

                # W_prime_phi_over_eps_Q
                W_prime_phi_over_eps_Q[loc_nds[loc_nd_row] * 2] += (
                    np.dot(
                        W_prime_phi_at_quad_pnt * basis_val_at_quad_pnt[loc_nd_row, :],
                        quad_wghts,
                    )
                    / self.epsilon
                    / Q_
                    * 0.25
                    * self.h**2
                )
            # Construct the matrix
        A_mat = csc_matrix(
            (VV[:marker], (II[:marker], JJ[:marker])),
            shape=(self.dof_phi_theta_num, self.dof_phi_theta_num),
        )

        rhs_q = q_ - 0.5 * np.dot(W_prime_phi_over_eps_Q, phi_theta_)
        r1 = spsolve(A_mat, rhs)
        r2 = spsolve(A_mat, W_prime_phi_over_eps_Q)
        q = (2.0 * rhs_q + np.dot(W_prime_phi_over_eps_Q, r1)) / (
            2.0 + np.dot(W_prime_phi_over_eps_Q, r2) * self.lambda_
        )
        phi_theta = r1 - self.lambda_ * q * r2

        return phi_theta, q

    def get_nd_ind(self, elem_ind_x: int, elem_ind_y: int, loc_nd):
        if (
            (elem_ind_x == 0 and loc_nd in [0, 2])
            or (elem_ind_x == self.N - 1 and loc_nd in [1, 3])
            or (elem_ind_y == 0 and loc_nd in [0, 1])
            or (elem_ind_y == self.N - 1 and loc_nd in [2, 3])
        ):
            return -1
        else:
            loc_nd_y, loc_nd_x = divmod(loc_nd, 2)
            return (
                (elem_ind_y + loc_nd_y - 1) * (self.N - 1) + elem_ind_x + loc_nd_x - 1
            )

    def get_next_u(self, phi_theta: np.ndarray, phi_theta_0: np.ndarray, n: int):
        max_data_len = self.N**2 * 64
        II = -np.ones(max_data_len, dtype=np.int32)
        JJ = -np.ones(max_data_len, dtype=np.int32)
        VV = np.zeros(max_data_len)

        rhs = np.zeros((self.dof_ela_num,))

        marker = 0
        for elem_ind_y, elem_ind_x in product(range(self.N), range(self.N)):
            loc_nds = [
                elem_ind_y * (self.N + 1) + elem_ind_x,
                elem_ind_y * (self.N + 1) + elem_ind_x + 1,
                (elem_ind_y + 1) * (self.N + 1) + elem_ind_x,
                (elem_ind_y + 1) * (self.N + 1) + elem_ind_x + 1,
            ]
            loc_phi = np.array([phi_theta[2 * loc_nd] for loc_nd in loc_nds])
            loc_theta = np.array([phi_theta[2 * loc_nd + 1] for loc_nd in loc_nds])
            loc_theta_0 = np.array([phi_theta_0[2 * loc_nd + 1] for loc_nd in loc_nds])
            phi_at_quad_pnt = loc_phi @ basis_val_at_quad_pnt

            k_phi_at_quad_pnt = self.k_vec(phi_at_quad_pnt)
            m_phi_at_quad_pnt = self.m(phi_at_quad_pnt)
            theta_theta_0_at_quad_pnt = (
                loc_theta - loc_theta_0
            ) @ basis_val_at_quad_pnt
            z_at_quad_pnt = self.kappa + (1.0 - k_phi_at_quad_pnt) * (1.0 - self.kappa)

            loc_u_source = np.array(
                [
                    self.u_source_func(
                        elem_ind_x * self.h, elem_ind_y * self.h, n * self.tau
                    ),
                    self.u_source_func(
                        (elem_ind_x + 1) * self.h, elem_ind_y * self.h, n * self.tau
                    ),
                    self.u_source_func(
                        elem_ind_x * self.h, (elem_ind_y + 1) * self.h, n * self.tau
                    ),
                    self.u_source_func(
                        (elem_ind_x + 1) * self.h,
                        (elem_ind_y + 1) * self.h,
                        n * self.tau,
                    ),
                ]
            )
            loc_ux_source = loc_u_source[:, 0]
            loc_uy_source = loc_u_source[:, 1]

            for loc_nd_row in range(N_V):
                nd_row = self.get_nd_ind(elem_ind_x, elem_ind_y, loc_nd_row)
                if nd_row >= 0:
                    for loc_nd_col in range(N_V):
                        nd_col = self.get_nd_ind(elem_ind_x, elem_ind_y, loc_nd_col)
                        if nd_col >= 0:
                            # A_{x, x}
                            II[marker] = nd_row * 2
                            JJ[marker] = nd_col * 2
                            VV[marker] = np.dot(
                                z_at_quad_pnt
                                * (
                                    2.0
                                    * self.mu
                                    * ela_mu_stiff_at_quad_pnt[
                                        loc_nd_row * 2, loc_nd_col * 2, :
                                    ]
                                    + self.lamb_3d
                                    * ela_lambda_stiff_at_quad_pnt[
                                        loc_nd_row * 2, loc_nd_col * 2, :
                                    ]
                                ),
                                quad_wghts,
                            )
                            marker += 1
                            # A_{x, y}
                            II[marker] = nd_row * 2
                            JJ[marker] = nd_col * 2 + 1
                            VV[marker] = np.dot(
                                z_at_quad_pnt
                                * (
                                    2.0
                                    * self.mu
                                    * ela_mu_stiff_at_quad_pnt[
                                        loc_nd_row * 2, loc_nd_col * 2 + 1, :
                                    ]
                                    + self.lamb_3d
                                    * ela_lambda_stiff_at_quad_pnt[
                                        loc_nd_row * 2, loc_nd_col * 2 + 1, :
                                    ]
                                ),
                                quad_wghts,
                            )
                            marker += 1
                            # A_{y, x}
                            II[marker] = nd_row * 2 + 1
                            JJ[marker] = nd_col * 2
                            VV[marker] += np.dot(
                                z_at_quad_pnt
                                * (
                                    2.0
                                    * self.mu
                                    * ela_mu_stiff_at_quad_pnt[
                                        loc_nd_row * 2 + 1, loc_nd_col * 2, :
                                    ]
                                    + self.lamb_3d
                                    * ela_lambda_stiff_at_quad_pnt[
                                        loc_nd_row * 2 + 1, loc_nd_col * 2, :
                                    ]
                                ),
                                quad_wghts,
                            )
                            marker += 1
                            # A_{y, y}
                            II[marker] = nd_row * 2 + 1
                            JJ[marker] = nd_col * 2 + 1
                            VV[marker] = np.dot(
                                z_at_quad_pnt
                                * (
                                    2.0
                                    * self.mu
                                    * ela_mu_stiff_at_quad_pnt[
                                        loc_nd_row * 2 + 1, loc_nd_col * 2 + 1, :
                                    ]
                                    + self.lamb_3d
                                    * ela_lambda_stiff_at_quad_pnt[
                                        loc_nd_row * 2 + 1, loc_nd_col * 2 + 1, :
                                    ]
                                ),
                                quad_wghts,
                            )
                            marker += 1
                    # Get loc rhs
                    # rhs_{x}
                    rhs[nd_row * 2] += (
                        np.dot(
                            z_at_quad_pnt
                            * (
                                m_phi_at_quad_pnt
                                - self.beta * theta_theta_0_at_quad_pnt
                            )
                            * 2.0
                            * (self.mu + self.lamb_3d)
                            * div_val_at_quad_pnt[loc_nd_row * 2, :],
                            quad_wghts,
                        )
                        * 0.5
                        * self.h
                    )
                    rhs[nd_row * 2] += (
                        np.dot(elem_bilinear_mass_mat[loc_nd_row, :], loc_ux_source)
                        * 0.25
                        * self.h**2
                    )
                    # rhs_{y}
                    rhs[nd_row * 2 + 1] += (
                        np.dot(
                            z_at_quad_pnt
                            * (
                                m_phi_at_quad_pnt
                                - self.beta * theta_theta_0_at_quad_pnt
                            )
                            * 2.0
                            * (self.mu + self.lamb_3d)
                            * div_val_at_quad_pnt[loc_nd_row * 2 + 1, :],
                            quad_wghts,
                        )
                        * 0.5
                        * self.h
                    )
                    rhs[nd_row * 2 + 1] += (
                        np.dot(elem_bilinear_mass_mat[loc_nd_row, :], loc_uy_source)
                        * 0.25
                        * self.h**2
                    )
        # Construct the matrix
        A_mat = csc_matrix(
            (VV[:marker], (II[:marker], JJ[:marker])),
            shape=(self.dof_ela_num, self.dof_ela_num),
        )
        u = spsolve(A_mat, rhs)
        return u
