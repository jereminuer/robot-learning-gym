import numpy as np
import torch
import scipy

############ DISCOUNTED SUM IMPLEMENTATION ###############
def discounted_sum(discount_factor, arr):
   """
   CREDIT: OpenAI spinning up implementation of PPO. Calculates cumalative sum:
      \ sum { \gamma^{t} * R_t }
   Don't know how it works. Can be found here: 
      https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py
   We use it to calculate expected return, and GAE Advantage
   """
   return scipy.signal.lfilter([1], [1, float(-discount_factor)], arr[::-1], axis=0)[::-1]

################# CUSTOM TRAJECTORY DATASET #################
class TrajectoryDataset():
   def __init__(self, trajectories: dict):
      self.trajectories = trajectories
   
   def __getitem__(self, index):
      experience = dict()
      for key, value in self.trajectories.items():
         experience[key] = value[index]
      return experience
   
   def __len__(self):
      return self.trajectories["advs"].size(0)

###################### REPLAY BUFFER ######################
class TrajectoryBuffer():
   def __init__(self, state_space, action_space, size=4000):
      """
      For every trajectory, in order to compute actor loss, we need the following:
         :param states = to calculate current distribution over states
         :param actions = to calculate current log prob of action over distribution
         :param prev_logprobs = constant denom in ratio term
         :param values = value estimation of states, used to calculate advantage
         :param advantages = previously calculated GAE advantages as constant scalar to gradient
         :param rewards = tracking reward to estimate returns
         :param returns = separate array to calculate returns, train value estimator on this

      To keep track of trajectories, we have:
         :param trajectory_start = index of start of trajectory in buffer. Trajectories are
         contiguous in the array
         :param idx = index in buffer for current step in trajectory.
         :param epoch_size = num timesteps after which we & perform update
      
      For GAE Advantage and discounted return, we have:
         :param lam = factor in cumlative sum for GAE
         :param gamma = discount for future returns
      """
      if (np.isscalar(state_space)):
         # Pass the shape as a tuple. Note that if we have to get nparray shape or tensor shape, also returns a tuple
         self.states = np.zeros((size, state_space), dtype=np.float32)
      else:
         # If state space is an iterable (dim1, dim2, ...), this will unpack it.
         self.states = np.zeros((size, *state_space), dtype=np.float32)
      if (np.isscalar(action_space)):
         self.actions = np.zeros((size, action_space), dtype=np.float32) 
      else:
         self.actions = np.zeros((size, *action_space), dtype=np.float32)

      self.prev_logprobs    = np.zeros(size, dtype=np.float32)
      self.vals             = np.zeros(size, dtype=np.float32)
      self.advantage        = np.zeros(size, dtype=np.float32)
      self.rewards          = np.zeros(size, dtype=np.float32)
      self.returns          = np.zeros(size, dtype=np.float32)

      self.trajectory_start, self.idx = 0, 0
      self.epoch_size = size
      self.lam = 0.9
      self.gamma = 0.99

   def clear(self):
      """
      Clears the buffer, so we can start building trajectories without worrying about
      complicating indices.
      """
      self.states           = np.zeros_like(self.states)
      self.actions          = np.zeros_like(self.actions)
      self.prev_logprobs    = np.zeros_like(self.prev_logprobs)
      self.vals             = np.zeros_like(self.vals)
      self.advantage        = np.zeros_like(self.advantage)
      self.rewards          = np.zeros_like(self.rewards)
      self.returns          = np.zeros_like(self.returns)

   def add_experience(self, s, a, prev_prob, vals, reward):
      """
      At each time step, we add the experience. Value of an action is estimated at this point.
      GAE Advantage will be calculated later using these values, and will scale the gradient.
      """
      self.states[self.idx]        = s
      self.actions[self.idx]       = a
      self.prev_logprobs[self.idx] = prev_prob
      self.vals[self.idx]          = vals
      self.rewards[self.idx]       = reward
      self.idx += 1
      # Note: we do NOT add returns or advantages, which are calculated at the end of trajectory

   def calculate_advantages(self, final_V=0):
      """
      At the end of a trajectory, go back and compute GAE estimated advantages (for actor loss)
      AND true discounted return (for critic loss)
      """
      sliced = slice(self.trajectory_start, self.idx)
      #print(f"TRAJECTORY START: {self.trajectory_start}")
      ##### CALCULATE GAE ######
      # (r_1, r_2, ... r_h)
      trajectory_rewards = np.append(self.rewards[sliced], final_V)
      # (V_1, V_2, ... V_h)
      values = np.append(self.vals[sliced], final_V)

      # r_t + gamma * V_{t+1} - V_t
      deltas = trajectory_rewards[:-1] + self.gamma * values[1:] - values[:-1]
      self.advantage[sliced] = discounted_sum(self.gamma * self.lam, deltas)


      ##### CALCULATE DISCOUNTED RETURN AT EACH TIME STEP ########
      trajectory_returns = discounted_sum(self.gamma, trajectory_rewards)[:-1]
      # We don't keep the last one?
      self.returns[sliced] = trajectory_returns

      self.trajectory_start = self.idx
   
   def get_trajectories(self):      
      """
      Once we fill up the entire buffer, return it all as one big batch to do training.
      Buffer will reset via zeroing indices -- which will overwrite previous values.
      """
      assert self.idx == self.epoch_size
      self.idx = 0
      self.trajectory_start = 0
      # Normalize the advantage values.
      adv_mean = self.advantage.mean()
      adv_std = self.advantage.std()
      self.advantage = (self.advantage - adv_mean) / (adv_std + 1e-8)
      # Normalize the return values -- DOESNT WORK
      # ret_mean = self.returns.mean()
      # ret_std = self.returns.std()
      # self.returns = (self.returns - ret_mean) / (ret_std + 1e-8)
      return dict(states        = torch.as_tensor(self.states, dtype=torch.float32), 
                  acts          = torch.as_tensor(self.actions, dtype=torch.float32), 
                  prev_logprobs = torch.as_tensor(self.prev_logprobs, dtype=torch.float32),
                  advs          = torch.as_tensor(self.advantage, dtype=torch.float32), 
                  rets          = torch.as_tensor(self.returns, dtype=torch.float32)
               )

   def get_trajectories_as_DataLoader(self, batch_size):
      """
      The purpose of this function is to do mini-batch stochastic gradient descent (SGD)
      instead of full-batch GD. Dictionary of trajectories is wrapped in custom dataset class.
      That is then wrapped in `torch.utils.data.DataLoader`. Allows variable batch sizes and shuffling.
      
         :param batch_size = mini batch size.
      """
      full_buffer_trajs = TrajectoryDataset(self.get_trajectories())
      return torch.utils.data.DataLoader(full_buffer_trajs, batch_size=batch_size, shuffle=True)