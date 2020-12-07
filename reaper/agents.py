import os, pickle
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from reaper.models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel, BilinearContrast
from reaper.utils import imagine_ahead, lambda_return, FreezeParameters, ActivateParameters, center_crop

class DreamerWorldModel(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config
    self.crop = self.config.crop_size if self.config.crop_size != self.config.render_size else None

    # Initialise model parameters  and networks randomly
    self.cropped_obs_size = (self.config.n_input_channels, self.config.crop_size, self.config.crop_size)
    # State transition model
    self.transition_model = TransitionModel(self.config.belief_size, self.config.state_size, self.config.env_action_size, self.config.hidden_size,
                                            self.config.embedding_size, self.config.dense_activation_function)
    # Observation encoder model
    self.encoder = Encoder(self.config.symbolic_env, self.cropped_obs_size, self.config.embedding_size, self.config.cnn_activation_function)
    # Observation reconstruction model
    self.observation_model = ObservationModel(self.config.symbolic_env, self.cropped_obs_size, self.config.belief_size, self.config.state_size,
                                         self.config.embedding_size, self.config.cnn_activation_function, )

    # Contrast-specific models
    if self.config.use_contrast_loss:
      self.key_encoder = Encoder(self.config.symbolic_env, self.cropped_obs_size, self.config.embedding_size, self.config.cnn_activation_function)
      self.contrast_model = BilinearContrast(self.config.embedding_size)

    # Reward prediction models
    self.reward_models = nn.ModuleList([RewardModel(self.config.belief_size, self.config.state_size, self.config.hidden_size, self.config.dense_activation_function,
                               ) for _ in range(self.config.n_tasks)])



    self.to(device=self.config.device)

  def update_belief(self, belief, posterior_state, action, observation):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    if self.crop is None:
      encoded_obs = self.encoder(observation).unsqueeze(dim=0)
    else:
      crop_obs = center_crop(observation,self.crop)
      encoded_obs = self.encoder(crop_obs).unsqueeze(dim=0)
    belief, prior_state, _, _, posterior_state, posterior_mean, posterior_std = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoded_obs)  # Action and observation need extra time dimension
    belief, posterior_state,prior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0) ,prior_state.squeeze(dim=0) # Remove time dimension from belief/state
    return belief, posterior_state, prior_state

  def get_reward(self, belief, posterior_state, actor_idx=0):
    return self.reward_models[actor_idx](belief,posterior_state)


class DreamerActor(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config
    # Policy model
    self.actor_model = ActorModel(self.config.belief_size, self.config.state_size, self.config.hidden_size, self.config.env_action_size,
                                  self.config.dense_activation_function,
                                  ).to(device=self.config.device)
    # Value model
    self.value_model = ValueModel(self.config.belief_size, self.config.state_size, self.config.hidden_size, self.config.dense_activation_function,
                                  ).to(device=self.config.device)
  def get_action(self, belief, posterior_state, explore=False):
    action = self.actor_model.get_action(belief, posterior_state, det=not(explore))
    if explore:
      action = torch.clamp(Normal(action, self.config.action_noise).rsample(), -1, 1) # Add gaussian exploration noise on top of the sampled action
    return action


class Dreamer():
  def __init__(self, config):
    super().__init__()
    self.config=config

    #Initialize world model trunk
    self.world_model = DreamerWorldModel(self.config)

    #Initialize actors
    self.actors=nn.ModuleList([DreamerActor(self.config) for _ in range(self.config.n_tasks)])

    #loss priors
    self.global_prior = Normal(torch.zeros(self.config.batch_size, self.config.state_size, device=self.config.device),
                          torch.ones(self.config.batch_size, self.config.state_size, device=self.config.device))  # Global prior N(0, I)
    self.free_nats = torch.full((1,), self.config.free_nats, device=self.config.device,
                           dtype=torch.float32)  # Allowed deviation in KL divergence

    #
    self.device = self.config.device

    #Optimizers
    self.model_optimizer = optim.Adam(self.world_model.parameters(), lr=0 if self.config.learning_rate_schedule != 0 else self.config.model_learning_rate, eps=self.config.adam_epsilon)
    self.actor_policy_optimizers = [optim.Adam(self.actors[_].actor_model.parameters(),
                                 lr=0 if self.config.learning_rate_schedule != 0 else self.config.actor_learning_rate,
                                 eps=self.config.adam_epsilon) for _ in range(self.config.n_tasks) ]
    self.actor_value_optimizers = [optim.Adam(self.actors[_].actor_model.parameters(),
                                 lr=0 if self.config.learning_rate_schedule != 0 else self.config.actor_learning_rate,
                                 eps=self.config.adam_epsilon) for _ in range(self.config.n_tasks) ]

  def step(self, belief, state, action, observation, env_idx=0, explore=False):
    with torch.no_grad():
      belief, posterior_state, prior_state = self.world_model.update_belief(belief, state, action, observation)
      action = self.actors[env_idx].get_action(belief, posterior_state, explore=explore)
    return action, belief, posterior_state, prior_state

  def get_model_loss(self,observations, actions, rewards, nonterminals, k_observations, env_idx=0, backprop=True):


      # Create initial belief and state for time t = 0
      init_belief, init_state = torch.zeros(self.config.batch_size, self.config.belief_size, device=self.device), torch.zeros(
        self.config.batch_size, self.config.state_size, device=self.config.device)
      # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
      encoded_obs = bottle(self.world_model.encoder, (observations[1:],))
      beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.world_model.transition_model(
        init_state, actions[:-1], init_belief, encoded_obs, nonterminals[:-1])

      ##################################################
      # MODEL OPTIMIZATION
      ##################################################

      # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting);
      # sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
      observation_full_loss = F.mse_loss(bottle(self.world_model.observation_model, (beliefs, posterior_states)), observations[1:],
                                         reduction='none').sum(dim=2 if self.config.symbolic_env else (2, 3, 4))
      observation_loss = observation_full_loss.mean(dim=(0, 1))

      reward_full_loss = F.mse_loss(bottle(self.world_model.reward_models[env_idx], (beliefs, posterior_states)),
                                    rewards[:-1], reduction='none')
      reward_loss = reward_full_loss.mean(dim=(0, 1))

      # transition loss
      div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
      kl_full_loss = torch.max(div, self.free_nats)
      kl_loss = kl_full_loss.mean(
        dim=(
        0, 1))


      # Computing batch model loss, full loss is used to update replay priority
      model_loss = self.config.observation_loss_scale * observation_loss + reward_loss + kl_loss
      sample_priority_value = self.config.observation_loss_scale * observation_full_loss[:-1] + reward_full_loss[
                                                                                                :-1] + kl_full_loss[:-1]

      if self.config.belief_l1_penalty > 0:
        model_loss += self.config.belief_l1_penalty * torch.abs(torch.cat([beliefs, posterior_states], dim=-1)).mean()

      # Contrastive representation learning
      contrast_loss=torch.zeros(1)
      if self.config.use_contrast_loss:
        encoded_pobs = bottle(self.world_model.key_encoder, (k_observations[1:],)).detach()
        contrast_logits = self.world_model.contrast_model(encoded_obs, encoded_pobs)
        contrast_labels = torch.LongTensor(np.arange(contrast_logits.shape[0])).to(self.config.device)
        contrast_full_loss = F.cross_entropy(contrast_logits, contrast_labels, reduction='none').view(-1,
                                                                                                      encoded_pobs.shape[
                                                                                                        1])
        contrast_loss = contrast_full_loss.mean()
        # Updating key parameters with exponential moving average
        for k_p, e_p in zip(self.world_model.key_encoder.parameters(), self.world_model.encoder.parameters()):
          k_p.data = k_p.data * (1 - 0.05) + e_p.data * (0.05)

        model_loss += self.config.contrast_loss_scale * contrast_loss
        sample_priority_value += self.config.contrast_loss_scale * contrast_full_loss[:-1]

      if backprop:
          # Update model parameters
          self.model_optimizer.zero_grad()
          model_loss.backward()
          nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip_norm, norm_type=2)
          self.model_optimizer.step()


      return model_loss, sample_priority_value.detach().cpu().numpy(), posterior_means, posterior_states, beliefs, kl_loss.item(), reward_loss.item(), observation_loss.item(), contrast_loss.item()

  def get_actor_loss(self, posterior_states, posterior_means, beliefs, env_idx=0, backprop=True):
    ##################################################
    # Policy OPTIMIZATION
    ##################################################
    actor_model = self.actors[env_idx].actor_model
    value_model = self.actors[env_idx].value_model
    actor_optimizer = self.actor_policy_optimizers[env_idx]
    value_optimizer = self.actor_value_optimizers[env_idx]

    # Dreamer implementation: actor loss calculation and optimization
    with torch.no_grad():
      actor_states = posterior_states.detach()
      actor_means = posterior_means.detach()
      actor_beliefs = beliefs.detach()
    with FreezeParameters(self.world_model.modules()):
      imagination_traj = imagine_ahead(actor_states, actor_beliefs, actor_model, self.world_model.transition_model,
                                       self.config.planning_horizon)

    # Imagine trajectories from policy starting on every observed state in the buffer
    imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs, imged_actions = imagination_traj
    with FreezeParameters(list(self.world_model.modules()) + value_model.modules):
      imged_reward = bottle(self.world_model.reward_models[env_idx], (imged_beliefs, imged_prior_states))
      value_pred = bottle(value_model, (imged_beliefs, imged_prior_states))

    returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=self.config.discount,
                            lambda_=self.config.disclam)
    actor_loss = -torch.mean(returns)
    # Update model parameters
    if backprop:
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_model.parameters(), self.config.grad_clip_norm, norm_type=2)
        actor_optimizer.step()

    # Dreamer implementation: value loss calculation and optimization
    with torch.no_grad():
      value_beliefs = imged_beliefs.detach()
      value_prior_states = imged_prior_states.detach()
      target_return = returns.detach()
    value_dist = Normal(bottle(value_model, (value_beliefs, value_prior_states)),
                        1)  # detach the input tensor from the transition network.
    value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
    # Update model parameters
    if backprop:
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), self.config.grad_clip_norm, norm_type=2)
        value_optimizer.step()
    return actor_loss.item(), value_loss.item()

  def compute_loss_and_train(self, observations, actions, rewards, nonterminals, k_observations, env_index, backprop=True):

    model_loss, sample_priority_value, posterior_means, posterior_states, beliefs, kl_loss, reward_loss, observation_loss, contrast_loss = self.get_model_loss(
        observations=observations, actions=actions, rewards=rewards, nonterminals=nonterminals, k_observations=k_observations,
        env_idx=env_index, backprop=backprop)
    actor_loss, value_loss = self.get_actor_loss(posterior_states=posterior_states,posterior_means=posterior_means, beliefs=beliefs, env_idx=env_index, backprop=backprop)

    return model_loss, kl_loss, reward_loss, observation_loss, contrast_loss, actor_loss, value_loss, sample_priority_value

  def load(self, path):
    model_dicts = torch.load(path, map_location=self.config.device)
    self.world_model.transition_model.load_state_dict(model_dicts['transition_model'])
    self.world_model.observation_model.load_state_dict(model_dicts['observation_model'])
    self.world_model.encoder.load_state_dict(model_dicts['encoder'])
    if self.config.use_contrast_loss:
      self.world_model.contrast_model.load_state_dict(model_dicts['contrast_model'])

    try:
      self.model_optimizer.load_state_dict(model_dicts['model_optimizer'])
    except KeyError:
      pass
    for _ in range(self.config.n_tasks):
      self.world_model.reward_models[_].load_state_dict(model_dicts['reward_model_{}'.format(_)])
      self.actors[_].actor_model.load_state_dict(model_dicts['actor_model_{}'.format(_)])
      self.actors[_].value_model.load_state_dict(model_dicts['value_model_{}'.format(_)])
      try:
        self.actor_policy_optimizers[_].load_state_dict(model_dicts['actor_optimizer_{}'.format(_)])
        self.actor_value_optimizers[_].load_state_dict(model_dicts['value_optimizer_{}'.format(_)])
      except KeyError:
        pass
    print('successful load')
  def save(self,path):
    torch_save_dict = {'transition_model': self.world_model.transition_model.state_dict(),
                     'observation_model': self.world_model.observation_model.state_dict(),
                     'encoder': self.world_model.encoder.state_dict(),
                     'model_optimizer': self.model_optimizer.state_dict()}
    if self.config.use_contrast_loss:
      torch_save_dict['contrast_model'] = self.world_model.contrast_model.state_dict()

    for _ in range(self.config.n_tasks):
      torch_save_dict['reward_model_{}'.format(_)] = self.world_model.reward_models[_].state_dict()
      torch_save_dict['actor_optimizer_{}'.format(_)] = self.actor_policy_optimizers[_].state_dict()
      torch_save_dict['value_optimizer_{}'.format(_)] = self.actor_value_optimizers[_].state_dict()
      torch_save_dict['actor_model_{}'.format(_)] = self.actors[_].actor_model.state_dict()
      torch_save_dict['value_model_{}'.format(_)] = self.actors[_].value_model.state_dict()


    torch.save(torch_save_dict, os.path.join(path))

  def train(self):
      self.world_model.train()
      for actor in self.actors:
          actor.train()

  def eval(self):
      self.world_model.eval()
      for actor in self.actors:
          actor.eval()