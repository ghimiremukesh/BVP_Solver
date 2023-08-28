'''
Contains frequently used utility scripts.
'''

import numpy as np
from scipy.io import loadmat
import os


def cheb(N):
    '''Build Chebyshev differentiation matrix.
    Uses algorithm on page 54 of Spectral Methods in MATLAB by Trefethen.'''
    theta = np.pi / N * np.arange(0, N + 1)
    X_nodes = np.cos(theta)

    X = np.tile(X_nodes, (N + 1, 1))
    X = X.T - X

    C = np.concatenate(([2.], np.ones(N - 1), [2.]))
    C[1::2] = -C[1::2]
    C = np.outer(C, 1. / C)

    D = C / (X + np.identity(N + 1))
    D = D - np.diag(D.sum(axis=1))

    # Clenshaw-Curtis weights
    # Uses algorithm on page 128 of Spectral Methods in MATLAB
    w = np.empty_like(X_nodes)
    v = np.ones(N - 1)
    for k in range(2, N, 2):
        v -= 2. * np.cos(k * theta[1:-1]) / (k ** 2 - 1)

    if N % 2 == 0:
        w[0] = 1. / (N ** 2 - 1)
        v -= np.cos(N * theta[1:-1]) / (N ** 2 - 1)
    else:
        w[0] = 1. / N ** 2

    w[-1] = w[0]
    w[1:-1] = 2. * v / N

    return X_nodes, D, w


def int_input(message, binary=True):
    while True:
        try:
            user_input = int(input(message))
            if binary and not (user_input == 0 or user_input == 1):
                raise TypeError
            break
        except ValueError:
            print("That doesn't seem to be an integer, try again...")
        except TypeError:
            print("That doesn't seem to be a 1 or 0, try again...")
    return user_input


def load_NN(model_path, return_stats=False):
    model_dict = loadmat(model_path)

    # parameters = {'weights1': model_dict['weights1'][0],
    #               'biases1': model_dict['biases1'][0],
    #               'weights2': model_dict['weights2'][0],
    #               'biases2': model_dict['biases2'][0]}

    parameters = {'weights': model_dict['weights'][0],
                  'biases': model_dict['biases'][0]}

    scaling = {'lb': model_dict['lb'], 'ub': model_dict['ub'],
               'A_lb': model_dict['A_lb'], 'A_ub': model_dict['A_ub'],
               'U_lb': model_dict['U_lb'], 'U_ub': model_dict['U_ub'],
               'V_min': model_dict['V_min'],
               'V_max': model_dict['V_max']}

    if return_stats:
        train_time = model_dict['train_time']
        val_grad_err = model_dict['val_grad_err']
        val_ctrl_err = model_dict['val_ctrl_err']
        return parameters, scaling, (train_time, val_grad_err, val_ctrl_err)
    else:
        return parameters, scaling


def softmax(x, alpha):
    return np.exp(alpha * x) / sum(np.exp(alpha * x))


def boltzmann_operator(x, alpha):
    return sum(x * np.exp(alpha * x)) / sum(np.exp(alpha * x))


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_target_status(state, beta):
    from choose_problem_swarm import problem
    """
    
    :param state: combined state of the system  
    :return: True if target reached
    """

    return (np.max(state[:problem.N_states] - state[problem.N_states:]) - beta >= 0)


def get_time_stats(states, ts, beta):
    """
    Given solution of BVP in the form of open loop states, return the stats of the time
    horizon

    :param states: states of the system
    :param beta: target ratio constant
    :return: T_min -- minimum time horizon, T_avg -- average time horizon, T_max -- max
             time horizon
    """
    Ts = []
    index = 0
    try:
        for traj in states:  # one trajectory out of all data
            target_reached = False
            count = 0
            while (not target_reached) and (count < len(traj.T) - 1):
                target_reached = check_target_status(traj.T[count], beta)
                count += 1

            if target_reached:
                Ts.append(ts[index][count])
            else:
                Ts.append(None)

            index += 1
    except:
        target_reached = False
        count = 0
        while (not target_reached) and (count < len(states.T) - 1):
            target_reached = check_target_status(states.T[count], beta)
            count += 1

        if target_reached:
            Ts.append(ts[count])
        else:
            Ts.append(None)

    return Ts
