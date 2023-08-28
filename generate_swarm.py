'''
This script generates data from scratch using time-marching.
'''
import os.path

import numpy as np

import utils.other
from utils.BVP_solver import solve_bvp
import scipy.io
import time
import warnings
import copy

from utils.other import int_input

from choose_problem_swarm import system, problem, config

from tqdm import tqdm

np.seterr(over='warn', divide='warn', invalid='warn')
warnings.filterwarnings('error')

# change the seed number you want
# np.random.seed(config.random_seeds['generate'])
N_states = problem.N_states
alpha = problem.alpha

# tags = ['na_na', 'na_a', 'a_a', 'a_na']
# idx = 0  # change the case you choose
#
# tag = tags[idx]
# if idx == 0:
#     problem.theta1 = 5
#     problem.theta2 = 5
# elif idx == 1:
#     problem.theta1 = 5
#     problem.theta2 = 1
# elif idx == 2:
#     problem.theta1 = 1
#     problem.theta2 = 1
# else:
#     problem.theta1 = 1
#     problem.theta2 = 5
#
# print(problem.theta1, problem.theta2)

# Validation or training data?
data_type = int_input('What kind of data? Enter 0 for validation, 1 for training:')
if data_type:
    data_type = 'train'
    # data_type = 'test'
else:
    data_type = 'val'

save_dir = 'examples/' + system
save_path = 'examples/' + system + '/data_' + data_type + '.mat'
print(save_path)
Ns = 10  # config.Ns[data_type]
X0_pool = problem.sample_X0(Ns)

t_OUT = np.empty((1, 0))
X_OUT = np.empty((2 * N_states, 0))
A_OUT = np.empty((2 * N_states, 0))
V_OUT = np.empty((1, 0))
# Tf_OUT = np.empty((1, 0))

N_sol = 0
sol_time = []
step = 0

N_converge = 0

time_horizon = problem.t1
print(time_horizon, problem.t1)

# ---------------------------------------------------------------------------- #
pbar = tqdm(total=Ns)
while N_sol < Ns:
    print('Solving BVP #', N_sol + 1, 'of', Ns, '...', end=' ')

    step += 1
    X0 = X0_pool[:, N_sol]
    print(step)
    print(X0)
    bc = problem.make_bc(X0)

    start_time = time.time()
    tol = 1e-3

    # Initial guess setting
    # X_guess = np.vstack((X0.reshape(-1, 1),
    #                      np.array([[alpha],
    #                                [alpha * time_horizon],  # alpha * 3.
    #                                [0.],
    #                                [0.],
    #                                [0.],
    #                                [0.],
    #                                [alpha],
    #                                [alpha * time_horizon],  # alpha * 3.
    #                                [0.],
    #                                [0.]])))

    X_guess = np.vstack((X0.reshape(-1, 1), 0 * np.ones((8, 1)), np.ones((1, 1))))  # initial guesses of all zeros

    X_guess1 = X_guess

    '''
    New guessing for X and t, go through all the actions and find the global solutions
    '''
    V_list = np.empty((1, 0))
    actions = [(np.zeros((4, )), np.zeros((4, ))), (np.zeros((4, )),  1 * np.ones((4, ))),
               (1 * np.ones((4, )), np.zeros((4, ))), (1 * np.ones((4, )), 1 * np.ones((4, )))]
    X_sol = []
    Y_sol = []
    rms_sol = []
    '''
    time horizon is 3 seconds
    '''
    t_guess1 = np.linspace(0, time_horizon, 4)
    dt = t_guess1 #np.diff(t_guess1)

    for i in range(3):
        X_guess1 = np.hstack((X_guess1, X_guess))
    for action in actions:
        u1, u2 = action
        for i in range(int(time_horizon)):
            # # state guess player 1
            # X_guess1[0, i + 1] = X_guess1[0, 0] + (u1[3] * X_guess1[3, 0] - u1[0] * X_guess1[0, 0]) * dt[i + 1]
            # X_guess1[1, i + 1] = X_guess1[1, 0] + (u1[0] * X_guess1[0, 0] - u1[1] * X_guess1[1, 0]) * dt[i + 1]
            # X_guess1[2, i + 1] = X_guess1[2, 0] + (u1[1] * X_guess1[1, 0] - u1[2] * X_guess1[2, 0]) * dt[i + 1]
            # X_guess1[3, i + 1] = X_guess1[3, 0] + (u1[2] * X_guess1[2, 0] - u1[3] * X_guess1[3, 0]) * dt[i + 1]
            X_guess1[0, i + 1] = X_guess1[0, i] + (1 - np.exp(-(u1[0] + u1[3]) * t_guess1[i + 1]))
            X_guess1[1, i + 1] = X_guess1[1, i] + (1 - np.exp(-(u1[0] + u1[1]) * t_guess1[i + 1]))
            X_guess1[2, i + 1] = X_guess1[2, i] + (1 - np.exp(-(u1[1] + u1[2]) * t_guess1[i + 1]))
            X_guess1[3, i + 1] = X_guess1[3, i] + (1 - np.exp(-(u1[2] + u1[3]) * t_guess1[i + 1]))
            # state guess player 2
            X_guess1[4, i + 1] = X_guess1[4, i] + (1 - np.exp(-(u2[0] + u2[3]) * t_guess1[i + 1]))
            X_guess1[5, i + 1] = X_guess1[5, i] + (1 - np.exp(-(u2[0] + u2[1]) * t_guess1[i + 1]))
            X_guess1[6, i + 1] = X_guess1[6, i] + (1 - np.exp(-(u2[1] + u2[2]) * t_guess1[i + 1]))
            X_guess1[7, i + 1] = X_guess1[7, i] + (1 - np.exp(-(u2[2] + u2[3]) * t_guess1[i + 1]))
            # costate guess for player 1
            X_guess1[8, i + 1] = X_guess1[8, 0] * np.exp(u1[0] * t_guess1[i + 1])
            X_guess1[9, i + 1] = X_guess1[9, 0] * np.exp(u1[1] * t_guess1[i + 1])
            X_guess1[10, i + 1] = X_guess1[10, 0] * np.exp(u1[2] * t_guess1[i + 1])
            X_guess1[11, i + 1] = X_guess1[11, 0] * np.exp(u1[3] * t_guess1[i + 1])
            # costate guess for player 2
            X_guess1[12, i + 1] = X_guess1[12, 0] * np.exp(u2[0] * t_guess1[i + 1])
            X_guess1[13, i + 1] = X_guess1[13, 0] * np.exp(u2[1] * t_guess1[i + 1])
            X_guess1[14, i + 1] = X_guess1[14, 0] * np.exp(u2[2] * t_guess1[i + 1])
            X_guess1[15, i + 1] = X_guess1[15, 0] * np.exp(u2[3] * t_guess1[i + 1])


        SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
                        verbose=0, tol=tol, max_nodes=5000)
        # try:
        #     SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
        #                     verbose=2, tol=tol, max_nodes=1500)
        # except Exception as e:
        #     continue

        max_rms = np.max(SOL.rms_residuals)
        if max_rms < tol:
            V_tmp = (SOL.y[-1:, 0:1])
            V_list = np.hstack((V_list, V_tmp))
            X_sol.append(SOL.x)
            Y_sol.append(SOL.y)
            rms_sol.append(SOL.rms_residuals)
        else:
            pass

    if X_sol == []:
        pass
    else:
        index = np.argmax(V_list)

        t = X_sol[index]
        X = Y_sol[index][:2 * N_states]
        A = Y_sol[index][2 * N_states:4 * N_states]
        V = Y_sol[index][-1:]
        # Tf = Y_sol[index][-1:]
        rms = np.max(rms_sol[index])

        sol_time.append(time.time() - start_time)

        t_OUT = np.hstack((t_OUT, t.reshape(1, -1)))
        X_OUT = np.hstack((X_OUT, X))
        A_OUT = np.hstack((A_OUT, A))
        V_OUT = np.hstack((V_OUT, V))
        # Tf_OUT = np.hstack((Tf_OUT, Tf))

        N_converge += 1

    N_sol += 1

    pbar.update(1)

sol_time = np.sum(sol_time)

print('')
print(step, '/', step, 'successful solution attempts:')
print('Average solution time: %1.1f' % (sol_time / step), 'sec')
print('Total solution time: %1.1f' % (sol_time), 'sec')

print('')
print('Total data generated:', X_OUT.shape[1])
print('Converge Number:', N_converge)
print('')

# ---------------------------------------------------------------------------- #
save_data = 1  # int_input('Save data? Enter 0 for no, 1 for yes:')
if save_data:
    if not os.path.exists(save_dir):
        utils.other.cond_mkdir(save_dir)
    try:
        save_dict = scipy.io.loadmat(save_path)

        overwrite_data = int_input('Overwrite existing data? Enter 0 for no, 1 for yes:')

        if overwrite_data:
            raise RuntimeWarning

        save_dict.update({'t': np.hstack((save_dict['t'], t_OUT)),
                          'X': np.hstack((save_dict['X'], X_OUT)),
                          'A': np.hstack((save_dict['A'], A_OUT)),
                          'V': np.hstack((save_dict['V'], V_OUT))})
                          # 'Tf': np.hstack((save_dict['Tf'], Tf_OUT))})

    except:
        U1, U2 = problem.U_star(np.vstack((X_OUT, A_OUT)))
        U = np.vstack((U1, U2))

        # save_dict = {'lb_1': np.min(X_OUT[:N_states], axis=1, keepdims=True),
        #              'ub_1': np.max(X_OUT[:N_states], axis=1, keepdims=True),
        #              'lb_2': np.min(X_OUT[N_states:2 * N_states], axis=1, keepdims=True),
        #              'ub_2': np.max(X_OUT[N_states:2 * N_states], axis=1, keepdims=True),
        #              'A_lb_11': np.min(A_OUT[:N_states], axis=1, keepdims=True),
        #              'A_ub_11': np.max(A_OUT[:N_states], axis=1, keepdims=True),
        #              'A_lb_12': np.min(A_OUT[N_states:2 * N_states], axis=1, keepdims=True),
        #              'A_ub_12': np.max(A_OUT[N_states:2 * N_states], axis=1, keepdims=True),
        #              'A_lb_21': np.min(A_OUT[2 * N_states:3 * N_states], axis=1, keepdims=True),
        #              'A_ub_21': np.max(A_OUT[2 * N_states:3 * N_states], axis=1, keepdims=True),
        #              'A_lb_22': np.min(A_OUT[3 * N_states:4 * N_states], axis=1, keepdims=True),
        #              'A_ub_22': np.max(A_OUT[3 * N_states:4 * N_states], axis=1, keepdims=True),
        #              'U_lb_1': np.min(U1, axis=1, keepdims=True),
        #              'U_ub_1': np.max(U1, axis=1, keepdims=True),
        #              'U_lb_2': np.min(U2, axis=1, keepdims=True),
        #              'U_ub_2': np.max(U2, axis=1, keepdims=True),
        #              'V_min_1': np.min(V_OUT[-2:-1, :]), 'V_max_1': np.max(V_OUT[-2:-1, :]),
        #              'V_min_2': np.min(V_OUT[-1, :]), 'V_max_2': np.max(V_OUT[-1, :]),
        #              't': t_OUT, 'X': X_OUT, 'A': A_OUT, 'V': V_OUT, 'U': U}
        save_dict = {'t': t_OUT, 'X': X_OUT, 'A': A_OUT, 'V': V_OUT, 'U': U}
        scipy.io.savemat(save_path, save_dict)
