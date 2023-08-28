##### Change this line for different a different problem. #####

# system = 'swarm'

system = 'swarm'

time_dependent = True  # True

if system == 'swarm':
    from problem_def_swarm import setup_problem, config_NN

if system == 'swarm-2d':
    from problem_def_swarm_2d import setup_problem, config_NN

problem = setup_problem()
config = config_NN(problem.N_states, time_dependent)
