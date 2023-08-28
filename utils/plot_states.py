import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import utils.other


def plot_(x1_values, x2_values, time_steps):
    # Define the bounding box sizes for each region
    box_sizes = [1, 1, 1, 1]

    # Multiply proportions by the total number of swarms
    total_swarms = 1000
    x1_counts = [[int(prop * total_swarms) for prop in timestep] for timestep in x1_values]
    x2_counts = [[int(prop * total_swarms) for prop in timestep] for timestep in x2_values]

    # Create subplots for each time step
    fig, axs = plt.subplots(len(time_steps), 2, figsize=(12, 4 * len(time_steps)))

    # Generate plots for each time step
    for t in range(len(time_steps)):
        # axs[t].set_xlim(-2, 2)
        # axs[t].set_ylim(-2, 2)
        # axs[t, 0].set_xlabel('X-axis')
        # axs[t, 0].set_ylabel('Y-axis')
        axs[t, 0].set_title(f'Time Step {t + 1}, t = {time_steps[t]}')
        axs[t, 0].xaxis.set_tick_params(labelbottom=False)
        axs[t, 0].yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        axs[t, 0].set_xticks([])
        axs[t, 0].set_yticks([])

        # Plot bounding boxes for each region
        corners = [(0, 0), (0, 2 * max(box_sizes)), (2 * max(box_sizes), 2 * max(box_sizes)), (2 * max(box_sizes), 0)]
        color1 = 'red'
        color2 = 'blue'

        for i, corner in enumerate(corners):
            rectangle = Rectangle(corner, box_sizes[i], box_sizes[i], edgecolor='black', facecolor='none')
            axs[t, 0].add_patch(rectangle)

            # Plot dots representing swarms in each region

            # alpha1 = x1_counts[t][i]/1000
            # alpha2 = x2_counts[t][i]/1000
            if x1_counts[t][i] > x2_counts[t][i]:
                alpha1 = 1
                alpha2 = 0.3
            else:
                alpha2 = 1
                alpha1 = 0.3
            x1_swarm_dots = np.random.uniform(corner[0], corner[0] + box_sizes[i], x1_counts[t][i])
            y1_swarm_dots = np.random.uniform(corner[1], corner[1] + box_sizes[i], x1_counts[t][i])
            axs[t, 0].scatter(x1_swarm_dots, y1_swarm_dots, color=color1, alpha=alpha1)

            x2_swarm_dots = np.random.uniform(corner[0], corner[0] + box_sizes[i], x2_counts[t][i])
            y2_swarm_dots = np.random.uniform(corner[1], corner[1] + box_sizes[i], x2_counts[t][i])
            axs[t, 0].scatter(x2_swarm_dots, y2_swarm_dots, color=color2, alpha=alpha2)

            # Add region labels
            if i == 0 or i == 3:  # regions 1 and 3
                axs[t, 0].annotate(f"Region {i + 1}", (corner[0] + box_sizes[i] / 2, corner[1] + 1.1 * box_sizes[i]),
                                   ha='center', va='center')
            else:
                axs[t, 0].annotate(f"Region {i + 1}", (corner[0] + box_sizes[i] / 2, corner[1] - box_sizes[i] / 4),
                                   ha='center', va='center')
        # Plot bar chart comparing the number of swarms in each region
        regions = ['Region 1', 'Region 2', 'Region 3', 'Region 4']
        counts_x1 = x1_counts[t]
        counts_x2 = x2_counts[t]
        bar_width = 0.35

        x = np.arange(len(regions))
        axs[t, 1].bar(x, counts_x1, width=bar_width, label='Attackers', color='red', alpha=0.5)
        axs[t, 1].bar(x + bar_width, counts_x2, width=bar_width, label='Defenders', color='blue', alpha=0.5)

        axs[t, 1].set_xticks(x + bar_width / 2)
        axs[t, 1].set_xticklabels(regions)
        axs[t, 1].set_ylabel('Number of Swarms')
        axs[t, 1].set_title('Number of Swarms in Each Region')
        axs[t, 1].legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import scipy.io as scio
    from choose_problem_swarm import system, problem, config

    data = scio.loadmat('../examples/swarm/data_test2.5s.mat')

    T = data['t']
    if len(T[0]) > 20:  # failsafe
        idxs = np.where(T == 0)[1]  # these are the start point of trajectories

        ts = [T.flatten()[idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
        ts.append(T.flatten()[idxs[-1]:])

        X = data['X']
        Xs = [X[:, idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
        Xs.append(X[:, idxs[-1]:])  # add the last remaining trajectory

        # Time horizon stats
        Times = utils.other.get_time_stats(Xs, ts, beta=0.8)

        U = data['U']
        Us = [U[:, idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]

        V = data['V']
        Vs = [V[:, idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]

        idx = 0

        t = ts[idx]
        Vs = Vs[idx]

        X_1 = Xs[idx] # pick 1 trajector and see
        X_1_1 = X_1[:problem.N_states, :].T
        X_1_2 = X_1[problem.N_states:, :].T

        U_1 = Us[idx]
        U_1_1 = U_1[:problem.N_states, :].T
        U_1_2 = U_1[problem.N_states:, :].T


        time_steps = t
    else:
        time_steps = T.squeeze()
        X = data['X']
        X_1_1 = X[:problem.N_states, :].T
        X_1_2 = X[problem.N_states:, :].T

        U = data['U']
        U_1_1 = U[:problem.N_states, :].T
        U_1_2 = U[problem.N_states:, :].T

        Vs = data['V']


    # Swarm proportions at different time steps
    # time_steps = 10
    # x1_values = [[0.4, 0.1, 0.2, 0.3],
    #              [0.3, 0.15, 0.25, 0.3],
    #              [0.2, 0.2, 0.3, 0.3],
    #              [0.35, 0.25, 0.2, 0.2],
    #              [0.1, 0.3, 0.25, 0.35],
    #              [0.25, 0.35, 0.25, 0.15],
    #              [0.3, 0.2, 0.3, 0.2],
    #              [0.2, 0.3, 0.2, 0.3],
    #              [0.15, 0.25, 0.35, 0.25],
    #              [0.3, 0.1, 0.4, 0.2]]
    #
    # x2_values = [[0.3, 0.2, 0.25, 0.25],
    #              [0.35, 0.25, 0.2, 0.2],
    #              [0.4, 0.1, 0.2, 0.3],
    #              [0.2, 0.3, 0.2, 0.3],
    #              [0.3, 0.1, 0.4, 0.2],
    #              [0.1, 0.4, 0.2, 0.3],
    #              [0.25, 0.3, 0.35, 0.1],
    #              [0.15, 0.35, 0.25, 0.25],
    #              [0.3, 0.3, 0.15, 0.25],
    #              [0.2, 0.2, 0.3, 0.3]]

    plot_(X_1_1, X_1_2, time_steps)

    print('Attackers action: ', U_1_1)
    print('Defenders action: ', U_1_2)
    print('Values: ', Vs)

    c_time = utils.other.get_time_stats(X, time_steps, 0.4)

    print("Capture occurs at t = ", c_time[0])
    # idxs = np.where(T == 0)[1]  # these are the start point of trajectories
    #
    # ts = [T.flatten()[idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
    #
    # ts.append(T.flatten()[idxs[-1]:])
    #
    # X = data['X']
    # Xs = [X[:, idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
    #
    # Xs.append(X[:, idxs[-1]:])  # add the last remaining trajectory
    #
    # # Time horizon stats
    # Times = utils.other.get_time_stats(Xs, ts, beta=0.4)
    #
    # U = data['U']
    # Us = [U[:, idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
    #
    # Us.append(U[:, idxs[-1]:])
    #
    # V = data['V']
    # Vs = [V[:, idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]
    #
    # Vs.append(V[:, idxs[-1]:])
    #
    # idx = 1
    #
    # t = ts[idx]
    # Vs = Vs[idx]
    #
    # X_1 = Xs[idx] # pick 1 trajector and see
    # X_1_1 = X_1[:problem.N_states, :].T
    # X_1_2 = X_1[problem.N_states:, :].T
    #
    # U_1 = Us[idx]
    # U_1_1 = U_1[:problem.N_states, :].T
    # U_1_2 = U_1[problem.N_states:, :].T
    #
    #
    # time_steps = t
    #
    #
    # # Swarm proportions at different time steps
    # # time_steps = 10
    # # x1_values = [[0.4, 0.1, 0.2, 0.3],
    # #              [0.3, 0.15, 0.25, 0.3],
    # #              [0.2, 0.2, 0.3, 0.3],
    # #              [0.35, 0.25, 0.2, 0.2],
    # #              [0.1, 0.3, 0.25, 0.35],
    # #              [0.25, 0.35, 0.25, 0.15],
    # #              [0.3, 0.2, 0.3, 0.2],
    # #              [0.2, 0.3, 0.2, 0.3],
    # #              [0.15, 0.25, 0.35, 0.25],
    # #              [0.3, 0.1, 0.4, 0.2]]
    # #
    # # x2_values = [[0.3, 0.2, 0.25, 0.25],
    # #              [0.35, 0.25, 0.2, 0.2],
    # #              [0.4, 0.1, 0.2, 0.3],
    # #              [0.2, 0.3, 0.2, 0.3],
    # #              [0.3, 0.1, 0.4, 0.2],
    # #              [0.1, 0.4, 0.2, 0.3],
    # #              [0.25, 0.3, 0.35, 0.1],
    # #              [0.15, 0.35, 0.25, 0.25],
    # #              [0.3, 0.3, 0.15, 0.25],
    # #              [0.2, 0.2, 0.3, 0.3]]
    #
    # plot_(X_1_1, X_1_2, time_steps)
    #
    # print('Attackers action: ', U_1_1)
    # print('Defenders action: ', U_1_2)
    #
    # print("Capture Time: ", Times[idx])
    # print('Values: ', Vs)



