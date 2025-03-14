"""
   Train a chosen model on some environment (custom, Gymnasium, etc.)
   At the end of a training run, results will be added to `training_results/`
   and the model will be saved.
   If applicable, the simulator will be rendered for manual inspection.
"""
import gymnasium as gym
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import json
import os
import sys
from algos import baseppo

assert len(sys.argv) == 2, "Correct Usage: python3 [script] [filename_to_save_info_to]"
filename = sys.argv[1]
   

env = gym.make('Humanoid-v5', terminate_when_unhealthy=True)

ppo = baseppo.ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], buffer_size=4096, action_range=0.4)
ppo.train(env=env, epochs=500)
torch.save(ppo.pi.state_dict(), "rl_models/inpnorm_clipact_slowlr_pi")
torch.save(ppo.v.state_dict(), "rl_models/inpnorm_clipact_slowlr_v")

######## PLOTTING REWARDS, LOSSES ###########
policy_loss_history = ppo.pi_losses[::ppo.pi_gradsteps]
critic_loss_history = ppo.v_losses[::ppo.v_gradsteps]
rewards = ppo.reward_tracking
kl_diverge = ppo.kl_divergence
# Create a figure and a set of subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
ax1.plot(policy_loss_history)
ax1.set_title('Actor (policy) Loss')
ax2.plot(critic_loss_history)
ax2.set_title('Critic (V) Loss')
ax3.plot(rewards)
ax3.set_title('Reward for each trajectory')
ax4.plot(kl_diverge)
ax4.set_title('KL Divergence for each ')

# Store this information if we want to look at it later
training_run_info = dict(
   actor_loss=policy_loss_history,
   critic_loss=critic_loss_history,
   reward=rewards
)
# Open file, dump dictionary as json file, etc. etc.
curdir = os.path.dirname(os.path.abspath(__file__))
with open(curdir + "/training_results/" + filename + ".json", "w+") as f:
   json.dump(training_run_info, f, indent=4)


# Show the plot
plt.tight_layout()
plt.show()


env = gym.make('Humanoid-v5', render_mode="human", terminate_when_unhealthy=True)
rewards = []
with torch.no_grad():
   while True:
      state, _ = env.reset()
      rewards.append(0)
      done = False
      episode_len = 0
      while not done:
         action = ppo.act(state)
         #action, _ = baseline_ppo.predict(state, deterministic=True)
         state_p, reward, done, _, _ = env.step(action)
         state = state_p
         rewards[-1] += reward
         #time.sleep(0.05)
         episode_len +=1
         if episode_len > 1000:
            break
      print(f"Trajectory ended. Overall reward: {rewards[-1]}")
