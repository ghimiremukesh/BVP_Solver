import numpy as np
import torch
from problem_def_template import config_prototype, problem_prototype
from utils.other import softmax, boltzmann_operator


class config_NN(config_prototype):
    def __init__(self, N_states, time_dependent):
        self.N_layers = 3
        self.N_neurons = 64
        self.layers = self.build_layers(N_states,
                                        time_dependent,
                                        self.N_layers,
                                        self.N_neurons)

        self.random_seeds = {'train': 1122, 'generate': 214}  # 1122

        self.ODE_solver = 'RK23' #'RK23'
        # Accuracy level of BVP data
        self.data_tol = 1e-2  # 1e-03
        # Max number of nodes to use in BVP
        self.max_nodes = 5000  # 800000
        # Time horizon
        self.t1 = 2.5

        # Time subintervals to use in time marching
        Nt = 10  # 10
        self.tseq = np.linspace(0., self.t1, Nt + 1)[1:]

        # Time step for integration and sampling
        self.dt = 1e-01
        # Standard deviation of measurement noise
        self.sigma = np.pi * 1e-02

        # Which dimensions to plot when predicting value function V(0,x)?
        # (unspecified dimensions are held at mean value)
        self.plotdims = [0, 3]

        # Number of training trajectories
        self.Ns = {'train': 150, 'val': 600, 'test': 1}

        ##### Options for training #####
        # Number of data points to use in first training rounds
        # Set to None to use whole data set
        self.batch_size = None  # 200

        # Maximum factor to increase data set size each round
        self.Ns_scale = 2
        # Number of candidate points to pick from when selecting large gradient
        # points during adaptive sampling
        self.Ns_cand = 2
        # Maximum size of batch size to use
        self.Ns_max = 8192

        # Convergence tolerance parameter (see paper)
        self.conv_tol = 1e-03

        # maximum and minimum number of training rounds
        self.max_rounds = 1
        self.min_rounds = 1

        # List or array of weights on gradient term, length = max_rounds
        self.weight_A = [10.]  # 1
        # List or array of weights on control learning term, not used in paper
        self.weight_U = [0.]  # 0.1

        # Dictionary of options to be passed to L-BFGS-B optimizer
        # Leave empty for default values
        self.BFGS_opts = {}


class setup_problem(problem_prototype):
    def __init__(self):
        self.N_states = 4
        self.t1 = 2.5

        # Parameter setting for the equation X_dot = Ax+Bu
        # self.A = np.array([[0, 1], [0, 0]])
        # self.B = np.array([[0], [1]])

        # Initial condition bounds (different initial setting)
        # Initial position is [15m, 20m]
        # Initial velocity is [18m/s, 25m/s]
        self.X0_lb = np.zeros((self.N_states,))
        self.X0_ub = np.ones((self.N_states,))
        # self.X0_lb = np.array([[15.], [18.], [15.], [18.]])
        # self.X0_ub = np.array([[20.], [25.], [20.], [25.]])

        # weight for terminal loss -- temperature parameter for the "smooth-max"
        self.alpha = 0.5  #
        self.u_bar = 1  # upper bound for player 1's action
        self.d_bar = 1  # upper bound for player 2's action

    def U_star(self, X_aug):
        '''Control as a function of the costate.'''
        Hu_1 = X_aug[2*self.N_states:3*self.N_states]
        Hu_2 = X_aug[3*self.N_states:4*self.N_states]
        dHdu = np.array([[Hu_1[1, :] - Hu_1[0, :]], [Hu_1[2, :] - Hu_1[1, :]],
                         [Hu_1[3, :] - Hu_1[2, :]], [Hu_1[0, :] - Hu_1[3, :]]]).reshape(4, -1)
        dHdd = np.array([[Hu_2[1, :] - Hu_2[0, :]], [Hu_2[2, :] - Hu_2[1, :]],
                         [Hu_2[3, :] - Hu_2[2, :]], [Hu_2[0, :] - Hu_2[3, :]]]).reshape(4, -1)
        U1 = np.zeros((Hu_1.shape))
        U2 = np.zeros((Hu_2.shape))
        # U1[np.where(dHdu <= 0)] = 2 # min control
        U1[np.where(dHdu > 0)] = self.u_bar
        # U2[np.where(dHdd <= 0)] = 2  # min control
        U2[np.where(dHdd < 0)] = self.d_bar  #  since p2 minimizes
        # U1 = [self.u_bar if x > 0 else 2 for x in Hu_1]  # [self.u_bar * (x > 0) for x in x1_aug]
        # U2 = [self.d_bar if x > 0 else 2 for x in Hu_2]  # self.d_bar * (Hu_2 > 0)

        return U1, U2

    # Boundary function for BVP
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:2 * self.N_states]
            XT = X_aug_T[:2 * self.N_states]
            AT = X_aug_T[2 * self.N_states:4 * self.N_states]
            VT = X_aug_T[4 * self.N_states:]

            # Boundary setting for lambda(T) when it is the final time T
            X1_T = XT[:self.N_states]
            X2_T = XT[self.N_states:]

            del_XT = X1_T - X2_T

            dF_dXT = softmax(del_XT, self.alpha) * (1 + self.alpha * (del_XT - boltzmann_operator(del_XT, self.alpha)))

            # dFdXT = np.concatenate((np.array([self.alpha]),
            #                         np.array([-2 * (XT[1] - 18)]),
            #                         np.array([0]),
            #                         np.array([0]),
            #                         np.array([0]),
            #                         np.array([0]),
            #                         np.array([self.alpha]),
            #                         np.array([-2 * (XT[3] - 18)])))

            dFdXT = np.concatenate((dF_dXT, -dF_dXT))

            # Terminal cost in the value function, see the new version of HJI equation
            F = boltzmann_operator(del_XT, self.alpha)

            return np.concatenate((X0 - X0_in, AT - dFdXT, VT - F))

        return bc

    # PMP equation for BVP
    def aug_dynamics(self, t, X_aug):
        '''Evaluation of the augmented dynamics at a vector of time instances

        X_aug contains X_1, X_2, A_1, A_2, V

        L_1, L_2 are the costates

        Returns: \dot{x}
        '''

        # Control as a function of the costate
        U1, U2 = self.U_star(X_aug)

        # State for each vehicle
        X1 = X_aug[:self.N_states]
        X2 = X_aug[self.N_states:2 * self.N_states]

        x1_1, x1_2, x1_3, x1_4, = X1[0], X1[1], X1[2], X1[3]
        x2_1, x2_2, x2_3, x2_4 = X2[0], X2[1], X2[2], X2[3]

        # State space function: X_dot = Ax+Bu
        dX1_dt = np.vstack(((U1[3] * x1_4 - U1[0] * x1_1), (U1[0] * x1_1 - U1[1] * x1_2), (U1[1] * x1_2 - U1[2] * x1_3),
                           U1[2] * x1_3 - U1[3] * x1_4))
        dX2_dt = np.vstack(((U2[3] * x2_4 - U2[0] * x2_1), (U2[0] * x2_1 - U2[1] * x2_2), (U2[1] * x2_2 - U2[2] * x2_3),
                           U2[2] * x2_3 - U2[3] * x2_4))

        # lambda in Hamiltonian equation
        A_1 = X_aug[2 * self.N_states:3 * self.N_states]
        A_2 = X_aug[3 * self.N_states:4 * self.N_states]

        # Jacobian of dynamic
        """
        Jac_X1 = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [-v1 * np.sin(ang1), v1 * np.cos(ang1), 0, 0],
                           [np.cos(ang1), np.sin(ang1), 0, 0]])

        Jac_X2 = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [-v2 * np.sin(ang2), v1 * np.cos(ang2), 0, 0],
                           [np.cos(ang2), np.sin(ang2), 0, 0]])
        """

        # lambda_dot in PMP equation
        dA1_dt = np.array([U1[0] * (A_1[0] - A_1[1]), U1[1] * (A_1[1] - A_1[2]), U1[2] * (A_1[2] - A_1[3]),
                           U1[3] * (A_1[3] - A_1[0])])
        dA2_dt = np.array([U2[0] * (A_2[0] - A_2[1]), U2[1] * (A_2[1] - A_2[2]), U2[2] * (A_2[2] - A_2[3]),
                           U2[3] * (A_2[3] - A_2[0])])

        dVdt = np.ones((len(t)))

        return np.vstack((dX1_dt, dX2_dt, dA1_dt, dA2_dt, dVdt))
