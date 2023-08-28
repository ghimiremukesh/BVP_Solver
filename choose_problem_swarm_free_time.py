##### Change this line for different a different problem. #####

system = 'swarm_free_time'

time_dependent = True  # True

if system == 'swarm_free_time':
    from problem_def_swarm_free_time import setup_problem, config_NN

problem = setup_problem()
config = config_NN(problem.N_states, time_dependent)
