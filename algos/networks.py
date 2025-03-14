import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal

def init_policy_network(mod):
   """
   Takes a torch module. Checks if it is a linear layer (we don't initialize activation functions)\n
   For the last layer, initializes to 100x smaller weights than the other layers.
   Uses `xavier_uniform_` distribution.
   """
   if isinstance(mod, nn.Linear):
      # We added this attribute for the last layer
      if hasattr(mod, "is_policy_output"):
         # 100x smaller than otherewise (1)
         nn.init.xavier_uniform_(mod.weight, 0.01) 
      else:
         nn.init.xavier_uniform_(mod.weight)
      
      if mod.bias is not None:
         nn.init.zeros_(mod.bias)


###################### POLICY CLASS ######################

class Policy(nn.Module):
   """
   A stochastic policy parameterized by a neural network. Outputs the mean of
   a gaussian for each action, with the STD initialized to e^(-0.5)
   """
   def __init__(self, num_features: int, num_actions: int, action_range: int):
      """
         :param num_features = size of observation space / num input features into network\n
         :param num_actions = size of actions space / num output features from network\n
         :param action range = absolute value of continuous action range
      """
      super().__init__()
      self.action_range = action_range
      self.log_std = torch.nn.Parameter(-0.5*torch.ones(num_actions))
      self.layers = nn.Sequential(
         nn.Linear(in_features=num_features, out_features=128),
         nn.Tanh(),
         nn.Linear(in_features=128, out_features=128),
         nn.Tanh(),
         nn.Linear(in_features=128, out_features=num_actions)
      )
      # Define an attribute of the last layer that signifies it as policy output
      # We will initialize this last layer with 100x smaller weights (https://arxiv.org/pdf/2006.05990)
      self.layers[-1].is_policy_output = True
      self.apply(init_policy_network)
   
   def _get_distribution(self, x):
      """
      Batch of states are passed into network. mus: [batch_size x num_actions]
      Normals: [batch_size]
      """
      mus = self.layers(x)
      return Normal(mus, torch.exp(self.log_std))
   
   def _get_log_probs(self, distr: Normal,raw_actions) -> torch.Tensor:
      """
      Returns the log probability of some action being chosen.
      Note that exp( log(pi'(a)) - log(pi(a)) ) = pi'(a) / pi(a), but is more numerically stable.
      Also: log( pi(a_1) * pi(a_2) * ... * pi(a_n) ) = log(pi(a_1)) + log(pi(a_2)) + ... for N dimensions
      of action vector.
      """
      ### NEW: u = tanh(x), where x is the raw action.
      #logP(u) = logP(x) - logtanh'(x)
      log_act = distr.log_prob(raw_actions)
      # Since we want the log prob of the true action, we need to add this correction
      correction = torch.log(self.action_range * (1 - torch.tanh(raw_actions)**2) + 1e-6) # Small constant for numerical stability
      return (log_act - correction).sum(axis=-1)
      ## OLD VERSION, we know this works:
      #return distr.log_prob(raw_actions).sum(axis=-1)
   
###################### VALUE FUNCTION CLASS ######################   

class Critic(nn.Module):
   def __init__(self, num_features):
      super().__init__()
      self.layers = nn.Sequential(
         nn.Linear(in_features=num_features, out_features=128),
         nn.Tanh(),
         nn.Linear(in_features=128, out_features=128),
         nn.Tanh(),
         nn.Linear(in_features=128, out_features=1)
      )
   
   def forward(self, batch):
      # values will be [batch_size x 1] -> one prediction for each batch
      values = self.layers(batch)
      # HOWEVER, when scaling the probabilities, we just want a tensor of [batch_size].
      # This ensures scalars are multiplied out to each example in batch.
      return values.squeeze(-1)
