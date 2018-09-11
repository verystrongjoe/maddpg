from experiments.networks import ActorNetwork, CriticNetwork
import  numpy as np
from copy import deepcopy
from collections import namedtuple

import torch as th
import torch.nn as nn
from torch.optim import Adam

Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards', 'terminal'))


class MyAgent:

    def __init__(self, env, actor, critic, memory, reward_factor, batch_n, agent_n, input_dim, action_dim, warm_up_episode_n=10):

        self.env = env
        # suppose that agent type is homogeneous which means those agents have same observation and action space itself
        if env.n > 0:
            observation_shape = env.observation_space[0].shape[0]
            action_space = env.action_space[0].n

        # self.actor = ActorNetwork(batch_n, agent_n, input_dim, action_dim)
        # self.critic = CriticNetwork(batch_n, agent_n, input_dim, action_dim, 1)
        self.actor = actor
        self.critic = critic

        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        self.agent_n = agent_n
        self.memory = memory
        self.recent_action = None
        self.recent_observation = None
        self.training = True
        self.episode_step = 0
        self.batch_size = batch_n
        self.warmup_episode_n = warm_up_episode_n
        self.reward_factor = reward_factor
        self.action_dim = action_dim

        self.GAMMA = 0.95
        self.tau = 0.01

        self.critic_optimizer = Adam(self.actor.parameters(), lr=0.001)
        self.actor_optimizer = Adam(self.critic.parameters(), lr=0.001)

        # if self.use_cuda:
        #         self.actor.cuda()
        #         self.critic.cuda()
        #         self.target_actor.cuda()
        #         self.target_critic.cuda()

    def backward(self, r, t=False):
        FloatTensor = th.cuda.FloatTensor # if self.use_cuda else th.FloatTensor

        # todo : consider memory interval?
        self.memory.append(self.recent_observation,
                           self.recent_action,
                           r, t, training=self.training)

        if not self.training:
            return

        if self.episode_step > self.warmup_episode_n:
            transitions = self.memory.sample(self.batch_size)
            assert len(transitions) == self.batch_size
            batch = Experience(*zip(*transitions))

            s0_batch = th.stack(batch.states).type(FloatTensor)
            a_batch = th.stack(batch.actions).type(FloatTensor)
            r_batch = th.stack(batch.rewards).type(FloatTensor)
            s1_batch = th.stack(batch.next_states).type(FloatTensor)
            t1_batch = th.stack(batch.terminal).type(FloatTensor)

            assert r_batch.shape == (self.batch_size, self.agent_n,)
            assert t1_batch.shape == (self.batch_size,)

            """
            ******** critic update  *********
            """
            # todo : bring actor and critic target model to calculate td error
            # get target q value
            target_actions = self.target_actor.forward(s1_batch)
            assert target_actions == (self.batch_size, self.agent_n, self.action_dim)

            target_q_values = self.target_critic.forward(s1_batch, target_actions)
            assert target_q_values == (self.batch_size, 1)  # here single q value

            discounted_r_batch = self.GAMMA * target_q_values
            assert discounted_r_batch.shape == (self.batch_size, 1)
            targets = (r_batch + discounted_r_batch).reshape(self.batch_size, 1)

            # get prediction q value
            pred_q_values = self.critic.forward(s0_batch, target_actions)

            # it must not affect target network but policy actor network
            loss_Q = nn.MSELoss()(pred_q_values, targets.detach())
            # loss_Q = th.mean_(loss_Q)
            loss_Q = th.mean(loss_Q)
            loss_Q.backward()

            self.critic_optimizer.zero_grad() # todo : i am not sure.
            self.critic_optimizer.step()

            """
            ********* actor update *********
            """
            p = self.actor(s0_batch)

            e = th.sum(p * th.log(p + 1e-10), axis=2)
            e = th.mean(e, axis=1)
            q = self.critic(s0_batch, p)
            loss = -th.mean(q)
            loss += 0.01 * e
            loss = th.mean(loss)

            loss.backward()

            self.soft_update(self.target_critic, self.critic, self.tau)
            self.soft_update(self.target_actor, self.actor, self.tau)

    def forward(self, o):
        o = self.memory.get_recent_state(o)  # 3 * 18
        # a = self.actor.forward(o[0])
        o = th.from_numpy(np.array(o)).type(th.FloatTensor)
        a = self.actor.forward(o)
        self.recent_action = a
        self.recent_observation = o
        return a

    def soft_update(target, source, t):
        for target_param, source_param in zip(target.parameters(),
                                              source.parameters()):
            target_param.data.copy_(
                (1 - t) * target_param.data + t * source_param.data)

