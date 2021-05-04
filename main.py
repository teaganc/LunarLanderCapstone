import numpy as np
from tqdm import tqdm

from rl_glue import RLGlue
from lunar_lander import LunarLanderEnvironment
from agent import Agent
from softmax import softmax


def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    rl_glue = RLGlue(environment, agent)
        
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"], 
                                 experiment_parameters["num_episodes"]))

    env_info = {}
    agent_info = agent_parameters

    for run in range(1, experiment_parameters["num_runs"]+1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        agent_info["network_pickle"] = "network500.pickle"
        env_info["seed"] = run
        env_info["render"] = True

        rl_glue.rl_init(agent_info, env_info)
        
        for episode in tqdm(range(1, experiment_parameters["num_episodes"]+1)):
            rl_glue.rl_episode(experiment_parameters["timeout"])
            
            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward

def run_model():
    print("")

def main():
	experiment_parameters = {
	    "num_runs" : 1,
	    "num_episodes" : 300,
	    "timeout" : 1000
	}

	environment_parameters = {}
	current_env = LunarLanderEnvironment

	agent_parameters = {
	    'network_config': {
		'state_dim': 8,
		'num_hidden_units': 1024,
		'num_actions': 4
	    },
	    'optimizer_config': {
		'step_size': 1e-3,
		'beta_m': 0.9, 
		'beta_v': 0.999,
		'epsilon': 1e-8
	    },
	    'replay_buffer_size': 50000,
	    'minibatch_sz': 8,
	    'num_replay_updates_per_step': 4,
	    'gamma': 0.99,
	    'tau': 0.001
	}
	current_agent = Agent

	run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)


main()
