from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np

# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
  return output

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
      h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
      kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
      stride = (stride, stride)

    if type(pad) is not tuple:
      pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def transpose_conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad =0 ):
  """
  Utility function for computing output of convolutions
  takes a tuple of (h,w) and returns a tuple of (h,w)
  """

  if type(h_w) is not tuple:
    h_w = (h_w, h_w)

  if type(kernel_size) is not tuple:
    kernel_size = (kernel_size, kernel_size)

  if type(stride) is not tuple:
    stride = (stride, stride)

  if type(pad) is not tuple:
    pad = (pad, pad)

  if type(dilation) is not tuple:
    dilation = (dilation, dilation)

  if type(out_pad) is not tuple:
      out_pad = (out_pad, out_pad)

  h = (h_w[0]-1)* stride[0] -2*pad[0] + dilation[0]*(kernel_size[0]-1) + out_pad[0] + 1
  w = (h_w[1]-1)* stride[1] -2*pad[1] + dilation[1]*(kernel_size[1]-1) + out_pad[1] + 1

  return h, w





class TransitionModel(jit.ScriptModule):
  __constants__ = ['min_std_dev']

  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.min_std_dev = min_std_dev
    self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
    self.rnn = nn.GRUCell(belief_size, belief_size)
    self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
    self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
    self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
    self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
    self.modules = [self.fc_embed_state_action, self.fc_embed_belief_prior, self.fc_state_prior, self.fc_embed_belief_posterior, self.fc_state_posterior]

  # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
  # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
  # t :  0  1  2  3  4  5
  # o :    -X--X--X--X--X-
  # a : -X--X--X--X--X-
  # n : -X--X--X--X--X-
  # pb: -X-
  # ps: -X-
  # b : -x--X--X--X--X--X-
  # s : -x--X--X--X--X--X-
  @jit.script_method
  def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None, detached_unroll:bool=False) -> List[torch.Tensor]:
    '''
    Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
    Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    '''
    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = actions.size(0) + 1
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
    # Loop over time sequence
    for t in range(T - 1):
      _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
      _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
      _state = _state.detach() if detached_unroll else _state # potentially detach gradients from successive states
      # Compute belief (deterministic hidden state)
      hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
      beliefs[t + 1] = self.rnn(hidden, beliefs[t])
      # Compute state prior by applying transition dynamics
      hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
      prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
      prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
      prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
      if observations is not None:
        # Compute state posterior by applying transition dynamics and using current observation
        t_ = t - 1  # Use t_ to deal with different time indexing for observations
        hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
        posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
        posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
        posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
    # Return new hidden states
    hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
    if observations is not None:
      hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
    return hidden


class SymbolicObservationModel(jit.ScriptModule):
  def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, observation_size)
    self.modules = [self.fc1, self.fc2, self.fc3]

  @jit.script_method
  def forward(self, belief, state):
    x = torch.cat([belief, state], dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    observation = self.fc3(hidden)
    return observation


class VisualObservationModel(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):#, use_belief=True):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
    self.modules = [self.fc1, self.conv1, self.conv2, self.conv3, self.conv4]

  @jit.script_method
  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.fc1(x)  # No nonlinearity here
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    observation = self.conv4(hidden)
    return observation

# Encoder and Observation model ignore cropped_obs_size, cropped obs size must be 64x64 for this configuration !!!!
def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
  else:
    return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):#, use_belief=True):
    # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    self.modules = [self.fc1, self.fc2, self.fc3]

  @jit.script_method
  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    reward = self.fc3(hidden).squeeze(dim=1)
    return reward

class ValueModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu',n_vals=1):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, n_vals)
    self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

  @jit.script_method
  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    reward = self.fc4(hidden).squeeze(dim=1)
    return reward

class ActorModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, action_size, dist='tanh_normal',
                activation_function='elu', min_std=1e-4, init_std=5, mean_scale=5):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.fc5 = nn.Linear(hidden_size, 2*action_size)
    self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std
    self._mean_scale = mean_scale

  @jit.script_method
  def forward(self, belief, state):# scale=torch.FloatTensor([1])):
    raw_init_std = torch.log(torch.exp(self._init_std) - 1)
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    hidden = self.act_fn(self.fc4(hidden))
    action = self.fc5(hidden).squeeze(dim=1)

    action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
    action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
    # action_std = scale*F.softplus(action_std_dev + raw_init_std) + self._min_std
    action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
    return action_mean, action_std

  # def get_action(self, belief, state, det=False,scale=None):# scale=torch.FloatTensor([1])):
  #   action_mean, action_std = self.forward(belief, state)
  #   # action_mean, action_std = self.forward(belief, state)#, scale)
  #   dist = Normal(action_mean, action_std)
  #   dist = TransformedDistribution(dist, TanhBijector())
  #   dist = torch.distributions.Independent(dist,1)
  #   dist = SampleDist(dist)
  #   if det: return dist.mode()
  #   else: return dist.rsample()
  def get_action(self, belief, state, det=False, scale=None):
    action_mean, action_std = self.forward(belief, state)
    if scale:
      #exploration distribution
      dist = Normal(action_mean, action_std + action_std.detach()*(1-scale))
      dist = TransformedDistribution(dist, TanhBijector())
      dist = torch.distributions.Independent(dist,1)
      dist = SampleDist(dist)
      action = dist.mode() if det else dist.rsample()
      proposal_loglike = dist.log_prob(action).detach()
      #true distribution
      dist = Normal(action_mean, action_std)
      dist = TransformedDistribution(dist, TanhBijector())
      dist = torch.distributions.Independent(dist,1)
      dist = SampleDist(dist)
      policy_loglike =dist.log_prob(action)
      return action, policy_loglike, proposal_loglike
    else:
      dist = Normal(action_mean, action_std)
      dist = TransformedDistribution(dist, TanhBijector())
      dist = torch.distributions.Independent(dist,1)
      dist = SampleDist(dist)
      action = dist.mode() if det else dist.rsample()
      return action

class MLPModel(jit.ScriptModule):
  def __init__(self, input_size, hidden_size, output_size, activation_function='relu' ):#, use_belief=True):
    # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)
    self.modules = [self.fc1, self.fc2, self.fc3 ]

  @jit.script_method
  def forward(self, x):

    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    output = self.fc3(hidden)
    return output


class SymbolicEncoder(jit.ScriptModule):
  def __init__(self, observation_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)
    self.modules = [self.fc1, self.fc2, self.fc3]

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.fc1(observation))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.fc3(hidden)
    return hidden


class VisualEncoder(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
    self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = hidden.view(-1, 1024)
    hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size, activation_function)
  else:
    return VisualEncoder(embedding_size, activation_function)

class BilinearContrast(jit.ScriptModule):
  def __init__(self, embedding_size ):#, use_belief=True):
    super().__init__()
    self.bilinear_W = Parameter(torch.Tensor(embedding_size, embedding_size))
    self.reset_parameters()

  def reset_parameters(self):
    init.kaiming_uniform_(self.bilinear_W, a=math.sqrt(5))

  @jit.script_method
  def forward(self, query,key):
    contrast_logits =  torch.matmul(
      torch.matmul(query.view(-1,query.shape[-1]), self.bilinear_W),
      key.view(-1,key.shape[-1]).transpose(0,1))
    contrast_logits = contrast_logits - contrast_logits.max(dim=1, keepdim=True)[0]
    return contrast_logits


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:
  def __init__(self, dist, samples=100):
    self._dist = dist
    self._samples = samples

  @property
  def name(self):
    return 'SampleDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    sample = dist.rsample()
    return torch.mean(sample, 0)

  def mode(self):
    dist = self._dist.expand((self._samples, *self._dist.batch_shape))
    sample = dist.rsample()
    logprob = dist.log_prob(sample)
    batch_size = sample.size(1)
    feature_size = sample.size(2)
    indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
    return torch.gather(sample, 0, indices).squeeze(0)

  def entropy(self):
    dist = self._dist.expand((self._samples, *self._dist.batch_shape))
    sample = dist.rsample()
    logprob = dist.log_prob(sample)
    return -torch.mean(logprob, 0)

  def sample(self):
    return self._dist.sample()
