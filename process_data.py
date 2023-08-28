import numpy as np
import scipy.io as scio
from choose_problem_swarm import system, problem, config
from utils import other

BETA = 0.8  # threshold for capture

data = scio.loadmat('examples/swarm-2d/data_train2.5s.mat')

T = data['t']
idxs = np.where(T == 0)[1]  # these are the start point of trajectories

ts = [T.flatten()[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]

X = data['X']
Xs = [X[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]

V = data['V']
Vs = [V[:, idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]

# Time horizon stats
Times = other.get_time_stats(Xs, ts, beta=BETA)

print("T_max is: ", np.max(Times))

# Now create a ground truth dataset with time-horizon of T_max
# Once the target is reached and t < T_max, fill in the ground truth with the last state and values

X_GT = []
V_GT = []

for i in range(len(Xs)):
    target_reached = False
    count = 0
    while (not target_reached) and (count < len(Xs[i].T)):
        target_reached = other.check_target_status(Xs[i].T[count], beta=BETA)
        X_GT.append(np.concatenate((ts[i][count].reshape(-1, ), Xs[i].T[count])))
        V_GT.append(Vs[i].squeeze()[count])
        count += 1
    if count < len(Xs[i].T):
        while (ts[i][count] <= 0.5) and (count < len(Xs[i].T)):
            X_GT.append(np.concatenate((ts[i][count].reshape(-1, ), Xs[i].T[count])))
            V_GT.append(Vs[i].squeeze()[count])
            count += 1


gt = {'states': np.vstack(X_GT),
      'values': np.vstack(V_GT)}

scio.savemat("ground_truth.mat", gt)






# U = data['U']
# Us = [U[:, idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]




