import tensorflow as tf
import numpy as np
import gym
import sys
import argparse
import matplotlib.pyplot as plt

class DuelingNetwork(object):

    def __init__(self, sess, n_actions, n_features, lr, gamma, replace_iter):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        # tf.set_random_seed(2)
        
        self.sess = sess
        self.replace_iter = replace_iter
        self.train_iter_counter = 0
        self.n_features = n_features
        self.n_actions = n_actions

        # build net
        self.s = tf.placeholder(tf.float32, [None, n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, n_features], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.done = tf.placeholder(tf.int32, [None, ], name='done')

        # current net   
        with tf.variable_scope('q_net_cur'):
            c1 = tf.layers.dense(inputs=self.s, units=16, activation=tf.nn.relu, name='c1')
            cv1 = tf.layers.dense(inputs=c1, units=32, activation=tf.nn.relu, name='cv1')
            cv2 = tf.layers.dense(inputs=cv1, units=1, name='cv2')
            ca1 = tf.layers.dense(inputs=c1, units=32, activation=tf.nn.relu, name='ca1')
            ca2 = tf.layers.dense(inputs=ca1, units=n_actions, name='ca2')                
            self.q_cur = cv2 + (ca2 - tf.reduce_mean(ca2, axis=1, keep_dims=True))

        # old net (using old fixed parameters)
        with tf.variable_scope('q_net_old'):
            o1 = tf.layers.dense(inputs=self.s_, units=16, activation=tf.nn.relu, name='o1')
            ov1 = tf.layers.dense(inputs=o1, units=32, activation=tf.nn.relu, name='ov1')
            ov2 = tf.layers.dense(inputs=ov1, units=1, name='ov2')
            oa1 = tf.layers.dense(inputs=o1, units=32, activation=tf.nn.relu, name='oa1')
            oa2 = tf.layers.dense(inputs=oa1, units=n_actions, name='oa2')                
            q_old = ov2 + (oa2 - tf.reduce_mean(oa2, axis=1, keep_dims=True))

        # use old network to compute target value
        with tf.variable_scope('q_target'):
            q_target = tf.stop_gradient(self.r + gamma * tf.reduce_max(q_old, axis=1, name='q_next_max') * tf.to_float(1 - self.done))

        with tf.variable_scope('q_cur_action'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            q_cur_action = tf.gather_nd(params=self.q_cur, indices=a_indices)

        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.squared_difference(q_target, q_cur_action))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)   

        # variables for updating old network      
        cur_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_net_cur')
        old_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_net_old')
        with tf.variable_scope('replace_old_net'):
            self.replace_op = [tf.assign(o, c) for o, c in zip(old_net_params, cur_net_params)]

        self.sess.run(tf.global_variables_initializer())

    def replace_old_net(self):
        # update old network
        self.sess.run(self.replace_op)

    def train(self, batch_memory):
        if self.train_iter_counter % self.replace_iter == 0:
            self.replace_old_net()

        self.sess.run(
            self.train_op,
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features+1],
                self.s_: batch_memory[:, self.n_features+2:-1],
                self.done: batch_memory[:, -1]
            })

        self.train_iter_counter += 1

    def predict(self, observations):
        return self.sess.run(self.q_cur, feed_dict={self.s: observations})

class Replay_Memory(object):

    def __init__(self, n_features, memory_size):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in transitions define the number of transitions that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, n_features*2+3))
        self.memory_counter = 0

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=batch_size)
        batch = self.memory[sample_index]
        return batch

    def append(self, s, a, r, s_, done):
        # Appends transition to the memory. 
        transition = np.hstack((s, a, r, s_, done))    
        append_index = self.memory_counter % self.memory_size
        self.memory[append_index, :] = transition
        self.memory_counter += 1

class DuelingAgent(object):

    def __init__(self, n_actions, n_features, lr, gamma, e_start, e_end, e_decay, replace_iter, memory_size, batch_size, ):

        self.epsilon = e_start
        self.e_end = e_end
        self.e_decay = e_decay
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_features = n_features
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.network = DuelingNetwork(
                            sess=self.sess,
                            n_actions=n_actions,
                            n_features=n_features,
                            lr=lr,
                            gamma=gamma,
                            replace_iter=replace_iter
                        )

        self.memory = Replay_Memory(n_features=n_features, memory_size=memory_size)

    def epsilon_greedy_policy(self, observation, epsilon=None):
        # Creating epsilon greedy probabilities to sample from. 
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.uniform() > epsilon:
            action_values = self.network.predict(np.expand_dims(observation, axis=0))[0]
            # print(action_values)
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.n_actions)  
        return action            

    def greedy_policy(self, observation):
        # Creating greedy policy for test time. 
        action_values = self.network.predict(np.expand_dims(observation, axis=0))[0]
        action = np.argmax(action_values)
        return action          

    def train(self, observation, action, reward, observation_, done):
        self.memory.append(observation, action, reward, observation_, done)
        batch_memory = self.memory.sample_batch(batch_size=self.batch_size)
        self.network.train(batch_memory)

        if self.epsilon > self.e_end:
            self.epsilon -= self.e_decay

    # def train(self):
    #     if self.burn_in > 0:
    #         self.burn_in_memory()

    #     print('Start training...')
    #     self.performance_hist = []
    #     self.episode = 0

    #     for self.episode in range(self.n_episode):
    #         observation = self.env.reset()
    #         total_reward = 0

    #         # eps_step = 0
    #         while True:

    #             # sample action
    #             action = self.epsilon_greedy_policy(observation, self.epsilon)
    #             observation_, reward, done, info = self.env.step(action)
    #             # eps_step += 1
    #             # true_done = done
    #             # if eps_step == 199:
    #             #     true_done = False
    #             # self.memory.append(observation, action, reward, observation_, true_done)
                
    #             # save to memory
    #             self.memory.append(observation, action, reward, observation_, done)

    #             # sample batch
    #             batch_memory = self.memory.sample_batch(batch_size=self.batch_size)
    #             self.Q_network.train(batch_memory)

    #             total_reward += reward
    #             observation = observation_

    #             # decay epsilon
    #             if self.epsilon > self.e_end:
    #                 self.epsilon -= self.e_decay

    #             if done:
    #                 # print('episode: ', self.episode, 'total reward: ', round(total_reward, 2), 'epsilon', round(self.epsilon, 2))
    #                 break

    #     print('Training done. Evaluting final model...')
    #     performance, _ = self.test(100, self.episode)
    #     self.performance_hist.append([self.episode, self.Q_network.train_iter_counter, performance])


    # def test(self, test_iter, episode_n, model_file=None):
    #     # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
    #     # Here you need to interact with the environment, irrespective of whether you are using a memory.
    #     print('---------------Testing---------------')
    #     print('Episode:     ', episode_n)
    #     print('Iteration:   ', self.Q_network.train_iter_counter)
        
    #     reward_rec = np.zeros(test_iter)


    #     for episode in range(test_iter):
    #         observation = self.test_env.reset()

    #         while True:
    #             if self.render:
    #                 self.test_env.render()
    #             # self.test_env.render()

    #             action = self.epsilon_greedy_policy(observation, 0.05)
    #             observation_, reward, done, info = self.test_env.step(action)
    #             reward_rec[episode] += reward
    #             observation = observation_

    #             if done:
    #                 break
    #     avg = np.average(reward_rec)
    #     std = np.std(reward_rec)

    #     print('Avg reward:  ', avg)
    #     print('std:         ', std)

    #     return avg, std

    # def plot_performance(self):
    #     performance_hist = np.array(self.performance_hist)
    #     plt.plot(performance_hist[:,1], performance_hist[:,2])
    #     plt.ylabel('Average performance')
    #     plt.xlabel('Iteration')
    #     plt.show()

