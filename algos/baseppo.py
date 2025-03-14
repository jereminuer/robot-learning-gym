import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networks
from networks import Policy
from networks import Critic
import buffers
from buffers import TrajectoryBuffer



###################### PPO CLASS ######################

class ActorCritic():
   def __init__(self, state_dim, action_dim, buffer_size, action_range, verbose=False):
      # Networks
      self.pi = Policy(state_dim, action_dim, action_range)
      self.v = Critic(state_dim)
      # Buffer
      self.buffer = TrajectoryBuffer(state_dim, action_dim, size=buffer_size)
      self.buffer_size = buffer_size
      # Tracking
      self.pi_losses = []
      self.v_losses = []
      self.reward_tracking = []
      self.kl_divergence = []
      # Hyperparameters
      self.ratio_clip = 0.2
      self.pi_optim = optim.Adam(self.pi.parameters(), lr=1e-4)
      self.critic_opt = optim.Adam(self.v.parameters(), lr=3e-4)
      #self.critic_opt = optim.AdamW(self.v.parameters(), lr=2e-4, weight_decay=0.01)
      self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_opt, 0.95)
      self.pi_gradsteps = 5
      self.v_gradsteps = 5
      self.minib_size = 64
      self.reward_scalar = 1
      self.kl_coeff = 0.1
      # So we know how to apply the tanh to restrict our actions
      self.action_range = action_range
      #Other
      self.verbose = verbose
      # Set up observation tracking to normalize observations
      self.init_obs_tracking(state_dim)
   
   def init_obs_tracking(self, state_dim):
      """
      Initialize a dictionary `self.inputs` to keep a running average 
      of the standard deviation and mean of all the observations we've seen.

         :param state_dim = state dimension, number of means / stds to keep track of
      """
      self.inputs = dict(
         m2=np.zeros(state_dim),
         means=np.zeros(state_dim),
         step=0
      )
   
   def normalize_observation(self, state) -> torch.Tensor:
      """
      Takes a state, and normalizes it to have a mean of 0 and STD of 1,
      according to a running average that it also updates.
      Uses Welford's algorithm to keep track of the running mean / STD.

         :param state = state to be normalized
      """
      assert state.shape == self.inputs["means"].shape

      # Calculate Mean
      self.inputs["step"] += 1
      delta1 = state - self.inputs["means"]
      self.inputs["means"] += delta1 / self.inputs["step"]

      # Calcualte STD
      delta2 = state - self.inputs["means"]
      self.inputs["m2"] += delta1 * delta2
      var = self.inputs["m2"] / (self.inputs["step"] - 1) if self.inputs["step"] > 1 else 0.0
      #std = np.maximum(np.sqrt(var), 1e-6)
      std = np.sqrt(var)

      # Return the normalized observation
      #normalized_state = (state - self.inputs["means"]) / std
      normalized_state = (state - self.inputs["means"]) / (std + 1e-8)
      return torch.as_tensor(normalized_state, dtype=torch.float32)


   def step(self, states):
      """
      We don't need any gradients when we perform a step. Trajectories currently collected
      are done under the "old" policy and probs are used simply as the denom in the ratio term.
      """
      with torch.no_grad():
         # Get pi distribution over state(s)
         distrs = self.pi._get_distribution(states)
         # Sample action(s) from pi
         raw_actions = distrs.sample()
         # squish actions to valid range
         actions = F.tanh(raw_actions)*self.action_range
         # Record log probs for later (training pi network)
         log_probs = self.pi._get_log_probs(distrs, raw_actions)
         # Current estimate of value function in order to train policy
         values = self.v(states)
      return raw_actions.numpy(), log_probs.numpy(), values.numpy(), actions.numpy()


   def compute_actor_loss(self, data: dict) -> torch.Tensor:
      """
      Assume data is a dict, with all states, actions, rewards, in a trajectory.
      Goal is to calculate gradient step and take gradient step for one trajectory.
      """
      # Pull out states, actions, advantage values from a trajectory
      states, actions, advantage, old_probs = data['states'], data['acts'], data['advs'], data['prev_logprobs']

      # 1. Calculate r = [ pi'(a|s) ] / [pi(a|s)]
      distrs = self.pi._get_distribution(states)
      #print(distrs.batch_shape, actions.shape)
      primed_log_probs = self.pi._get_log_probs(distrs, actions)
      #print(f'ACTIONS: {actions}\n\n')
      #print(f'CURRENT LOG PROBS OF GIVEN ACTION (ones): {primed_log_probs}\n\n')
      #print(f'OLD LOG PROBS OF GIVEN ACTION (ones) - should be the same under same policy\n{old_probs}\n\n')
      # exp( log(pi'(a|s)) - log(pi(a|s)) )
      ratio = torch.exp(primed_log_probs - old_probs)
      # H(p) = - log(p(x))
      approx_kl_divergence = old_probs - primed_log_probs
      kl_loss = approx_kl_divergence * self.kl_coeff
      #print(f"RATIO: {ratio}")
      # 2. rclip(r) * A
      clipped_ratio = torch.clamp(ratio, 1-self.ratio_clip, 1+self.ratio_clip)
      # 3. If the advantage is negative, we don't clip. 
      # Note that we perform gradient ascent (but actually we just take negative and do gradient descent)
      surrogate_loss = -torch.min(clipped_ratio*advantage, ratio*advantage)
      #surrogate_loss = -torch.min(clipped_ratio*advantage, ratio*advantage) + kl_loss
      return surrogate_loss.mean(), approx_kl_divergence.mean().item()
      


   def compute_critic_loss(self, data: dict) -> torch.Tensor:
      """
      Critic Loss is very simple: (V(s) - returns(s))^2.
      Critic tries to exactly predict the returns for the rest of the 
      """
      states, returns = data['states'], data['rets']
      loss = torch.pow((self.v(states) - returns), 2).mean()
      return loss

   def act(self, state):
      """
      Calls ActorCritic.step(), and just throws away the log probs and values.
      Used for inference.
      """
      state = self.normalize_observation(state)
      return self.step(torch.as_tensor(state, dtype=torch.float32))[0]


   def gradient_step(self):
      """
      After completely filling the buffer, this function does mini-batch SGD over the entire
      buffer, for both the actor and the critic, `self.pi_gradsteps` and `self.v_gradsteps` times,
      respectively.
      """
      # These trajectories are wrapped by the `torch.utils.data.Dataloader` class
      trajectories = self.buffer.get_trajectories_as_DataLoader(batch_size=self.minib_size)

      # N full passes of buffer on critic
      for _ in range(self.v_gradsteps):
         # Track average loss over buffer and keep for logging purpose
         loss = []
         # mini batch SGD over entire buffer
         for minib_traj in trajectories:
            self.critic_opt.zero_grad()
            critic_loss = self.compute_critic_loss(minib_traj)
            critic_loss.backward()
            self.critic_opt.step()
            loss.append(critic_loss.item())
         
         self.v_losses.append(np.mean(loss))

      #N full passes of buffer on actor
      for s in range(self.pi_gradsteps):
         # Track average loss & kl divergence over buffer and keep for logging purpose
         loss = []
         kl_diverge = []
         #mini batch SGD over entire buffer
         for minib_traj in trajectories:
            self.pi_optim.zero_grad()
            actor_loss, dkl = self.compute_actor_loss(minib_traj)
            actor_loss.backward()
            self.pi_optim.step()
            # Tracking
            loss.append(actor_loss.item())
            kl_diverge.append(dkl)
         
         # Logging
         self.pi_losses.append(np.mean(loss))
         self.kl_divergence.append(np.mean(kl_diverge))
         if self.kl_divergence[-1] > 0.015:
            print(f"HIGH AVERAGE DIVERGENCE ON {s}: {self.kl_divergence[-1]}")
   
   def train(self, env, epochs):
      for i in range(epochs):
         state, _ = env.reset()
         # Track the trajectory rewards over each epoch
         trajectory_rewards = []
         trajectory_reward = 0.0

         #Scheduled learning rate for critic
         if i > 200:
            self.critic_scheduler.step()
         
         # Step function learning drop for actor -- uncertain whether to keep
         # if i == 300:
         #    for group in self.pi_optim.param_groups:
         #       group["lr"] /= 2
         #       self.ratio_clip /= 2
         
         # Cap maximum trajectory length for more diverse data in buffer
         step = 0

         for _ in range(self.buffer_size):
            # Normalize the observation
            state = self.normalize_observation(state)
            # Sample action, calculate value, log prob
            raw_action, log_prob, value, true_action = self.step(torch.as_tensor(state, dtype=torch.float32))
            state_p, reward, done, _,  _ = env.step(true_action)
            step += 1

            scaled_reward = reward * self.reward_scalar
            self.buffer.add_experience(state, raw_action, log_prob, value, scaled_reward)

            trajectory_reward += reward

            if done:
               # At each trajectory, we look back over the entire trajectory and calculate advantages, returns.
               self.buffer.calculate_advantages(final_V=0)

               # Store trajectory reward and reset
               if self.verbose: print(f"trajectory reward: {trajectory_reward}")
               trajectory_rewards.append(trajectory_reward)
               trajectory_reward = 0.0
               state, _ = env.reset()

            # Cap maximum trajectory length for more diverse data in buffer
            elif step >= 500:
               # Bootstrap the return with V(s) for normalized observation
               state_pnorm = self.normalize_observation(state_p)
               bootstrapped_return = self.v(state_pnorm).detach()
               self.buffer.calculate_advantages(bootstrapped_return)
               # Log the reward
               trajectory_rewards.append(trajectory_reward)
               # Reset the environment
               step = 0
               trajectory_reward = 0.0
               state, _ = env.reset()

            else:
               state = state_p

         # When we exit an epoch -- either we perfectly ended a trajectory (unlikely) or
         # we were in the middle. If so, need to calculate the advantages.
         if not done:
            state_pnorm = self.normalize_observation(state_p)
            #final_state_v = self.v(torch.as_tensor(state_p, dtype=torch.float32)).detach()
            final_state_v = self.v(state_pnorm).detach()
            self.buffer.calculate_advantages(final_state_v)
            trajectory_rewards.append(trajectory_reward)
         self.gradient_step()
         # Average the trajectory rewards across the epoch, store, reset list
         self.reward_tracking.append(np.mean(trajectory_rewards))
         print(f"TRAJECTORY REWARDS FOR EPOCH {i}: {trajectory_rewards}\n")
         print(f"ACTOR LOSSES FOR EPOCH {i}. BEFORE: {self.pi_losses[-5]}\tAFTER: {self.pi_losses[-1]}\n")
         print(f"CRITIC LOSSES FOR EPOCH {i}. BEFORE: {self.v_losses[-5]}\tAFTER: {self.v_losses[-1]}\n\n")