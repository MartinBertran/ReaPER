import cv2
import numpy as np
import torch


GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']
CONTROL_SUITE_ENVS = [
  'cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run', 'ball_in_cup-catch', 'walker-walk','reacher-hard', 'walker-run', 'humanoid-stand', 'humanoid-walk', 'fish-swim', 'acrobot-swingup',
'cartpole-two_poles','cartpole-three_poles','dog-stand','dog-walk','dog-trot','dog-run','dog-fetch',
'finger-turn_easy','finger-turn_hard','fish-upright','fish-swim',
'hopper-stand','hopper-hop','humanoid-run','manipulator-bring_ball','manipulator-bring_peg', 'point_mass-easy',
'manipulator-insert_ball','manipulator-insert_peg','pendulum-swingup',
'quadruped-walk','quadruped-run','quadruped-escape','quadruped-fetch',
'stacker-stack_2','stacker-stack_4','swimmer-swimmer6','swimmer-swimmer15','walker-stand',
]
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2, 'humanoid': 2, 'fish': 2, 'acrobot':4}


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

def rgb2rgbgr(rgb):
  rgb = rgb.astype('float32')
  gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])[...,np.newaxis]
  rgbgr = np.concatenate([gray,rgb],axis=-1)
  return rgbgr

def _images_to_observation(images, bit_depth, render_size=64, use_rgbgr=False):
  if use_rgbgr:
    images = torch.tensor(
      rgb2rgbgr(cv2.resize(images, (render_size, render_size), interpolation=cv2.INTER_LINEAR)).transpose(2, 0, 1),
      dtype=torch.float32)  # Resize and put channel first
  else:
    images = torch.tensor(cv2.resize(images, (render_size, render_size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension


class ControlSuiteEnv():
  def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, action_noise_scale=None, render_size=64, use_rgbgr=False):
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
    from dm_control.suite.wrappers import action_noise
    domain, task = env.split('-')
    self.symbolic = symbolic
    self.render_size= render_size if render_size else 64
    self.use_rgbgr= use_rgbgr
    if self.use_rgbgr:
      self.obs_tuple = (4,render_size,render_size)
    else:
      self.obs_tuple = (3, render_size, render_size)
    self.action_noise_scale = action_noise_scale
    self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
    if not symbolic:
      self._env = pixels.Wrapper(self._env)
    if self.action_noise_scale is not None:
      self._env = action_noise.Wrapper(self._env, scale=self.action_noise_scale)
    self.max_episode_length = max_episode_length
    if action_repeat < 0:
      try:
        action_repeat = CONTROL_SUITE_ACTION_REPEATS[domain]
      except KeyError:
        action_repeat = 2
    self.action_repeat = action_repeat
    try:
      if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
        print('Using action repeat %d; recommended action repeat for domain is %d' % (action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
    except KeyError:
      pass
    self.bit_depth = bit_depth

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    if self.symbolic:
      return torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32).unsqueeze(dim=0)
    else:
      return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth, self.render_size, use_rgbgr=self.use_rgbgr)

  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state = self._env.step(action)
      reward += state.reward if state.reward else 0.0  #state.reward
      self.t += 1  # Increment internal timer
      done = state.last() or self.t == self.max_episode_length
      if done:
        break
    if self.symbolic:
      observation = torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32).unsqueeze(dim=0)
    else:
      observation = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth, self.render_size, use_rgbgr=self.use_rgbgr)
    return observation, reward, done

  def render(self):
    cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
    self._env.close()

  @property
  def observation_size(self):
    return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()]) if self.symbolic else self.obs_tuple

  @property
  def action_size(self):
    return self._env.action_spec().shape[0]

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    spec = self._env.action_spec()
    return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))



class GymEnv():
  def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, use_rgbgr=False):
    import gym
    self.symbolic = symbolic
    self.use_rgbgr = use_rgbgr
    self._env = gym.make(env)
    self._env.seed(seed)
    self.max_episode_length = max_episode_length
    if action_repeat < 0:
      action_repeat = 1
    self.action_repeat = action_repeat
    self.bit_depth = bit_depth

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    if self.symbolic:
      return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth, use_rgbgr=self.use_rgbgr)
  
  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state, reward_k, done, _ = self._env.step(action)
      reward += reward_k
      self.t += 1  # Increment internal timer
      done = done or self.t == self.max_episode_length
      if done:
        break
    if self.symbolic:
      observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      observation = _images_to_observation(self._env.render(mode='rgb_array'), use_rgbgr=self.use_rgbgr)
    return observation, reward, done

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def observation_size(self):
    return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    return torch.from_numpy(self._env.action_space.sample())


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth,action_noise_scale=None, render_size=None, use_rgbgr=False):
  if env in GYM_ENVS:
    return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, use_rgbgr=use_rgbgr)
  elif env in CONTROL_SUITE_ENVS:
    return ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth,action_noise_scale, render_size=render_size, use_rgbgr=use_rgbgr)


# Wrapper for batching environments together
class EnvBatcher():
  def __init__(self, env_class, env_args, env_kwargs, n):
    self.n = n
    self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
    self.dones = [True] * n

  # Resets every environment and returns observation
  def reset(self):
    observations = [env.reset() for env in self.envs]
    self.dones = [False] * self.n
    return torch.cat(observations)

 # Steps/resets every environment and returns (observation, reward, done)
  def step(self, actions):
    done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]  # Done mask to blank out observations and zero rewards for previously terminated environments
    observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
    dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]  # Env should remain terminated if previously terminated
    self.dones = dones
    observations, rewards, dones = torch.cat(observations), torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones, dtype=torch.uint8)
    observations[done_mask] = 0
    rewards[done_mask] = 0
    return observations, rewards, dones

  def close(self):
    [env.close() for env in self.envs]

  def sample_random_action(self):
    return [env.sample_random_action() for env in self.envs]
