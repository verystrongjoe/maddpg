import torch
import argparse
import numpy as np
from experiments.memory import SequentialMemory
from experiments.agent import MyAgent
import time
import pickle
import keras

import maddpg.common.tf_util as U
# from maddpg.trainer.maddpg import MADDPGAgentTrainer
from experiments.networks import ActorNetwork
from experiments.networks import CriticNetwork

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=80000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")

    # Checkpoint
    parser.add_argument("--warm-up-episodes", type=int, default=2000, help="warm up episodes")

    parser.add_argument("--exp-name", type=str, default="default", help="name of the experiment")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")

    parser.add_argument("--save-dir", type=str, default="/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-critic-dir", type=str, default="tmp/policy", help="directory in which training state and model are saved")
    parser.add_argument("--save-actor-dir", type=str, default="tmp/policy", help="directory in which training state and model are saved")

    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    parser.add_argument("--load-critic-dir", type=str, default="tmp/policy", help="directory in which training state and model are loaded")
    parser.add_argument("--load-actor-dir", type=str, default="tmp/policy", help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(scenario_name, world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(scenario_name, world, scenario.reset_world, scenario.reward, scenario.observation)

    # additional setting
    # env.discrete_action_input = True

    return env


"""
just replace existing training logic to my model's one with pytorch 

simple snippet 

compile check
  (action repetition)
while Episode
  forward
  step based on action from above
  reward += r
  if max step, done 
  backward
"""
def train(arglist):

    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    obs_n = env.reset()

    rb = SequentialMemory(limit=100000, window_length=1)

    actor = ActorNetwork(32, env.n, env.observation_space[0].shape[0],
                         env.action_space[0].n, 32)
    critic = CriticNetwork(32, env.n,
                           env.observation_space[0].shape[0],
                           env.action_space[0].n, 1)

    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    if arglist.display or arglist.restore or arglist.benchmark:
        print('Loading previous state...')
        # actor = torch.load(arglist.load_actor_dir)
        # critic = torch.load(arglist.load_critic_dir)

    # todo : use agent
    agent = MyAgent(env, actor, critic, rb, 0.1, 32, env.n, env.observation_space, env.action_space, arglist.warm_up_episodes)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')

    while True:
        # get action
        action_n = agent.forward(obs_n)

        # sample action from probabilities
        # action_n = [keras.utils.to_categorical(a.detach(), 5) for a in action_n[0]]
        action_n = [a.detach() for a in action_n[0]]

        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # collect experience
        # agent.memory.append(obs_n, action_n, rew_n, done or terminal, True)
        # print(rew_n)

        # todo : each agent gets same reward as a one unified reward
        episode_rewards[-1] += rew_n[0]

        # todo : warm up
        agent.backward(rew_n[0], t=done or terminal)

        if done or terminal:

            agent.forward(obs_n)
            agent.backward(0., t=False)

            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            # for a in agent_rewards:
            #     a.append(0)
            # agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # save model, display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            # todo : pytorch save
            # torch.save(critic, arglist.save_critic_dir)
            # torch.save(actor, arglist.save_actor_dir)

            print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".
                format(train_step, len(episode_rewards),
                       np.mean(episode_rewards[-arglist.save_rate:]),
                       round(time.time()-t_start, 3)))

            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
            rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
