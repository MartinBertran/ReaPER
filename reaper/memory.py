import numpy as np
import torch
from reaper.env import postprocess_observation, preprocess_observation_
import random
from reaper.utils import SumTree, EpisodeSumTree, random_crop


class ExperienceReplay():
  def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device):
    self.device = device
    self.symbolic_env = symbolic_env
    self.size = size
    self.observations = np.empty((size, observation_size) if symbolic_env else (size, 3, 64, 64), dtype=np.float32 if symbolic_env else np.uint8)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32) 
    self.nonterminals = np.empty((size, 1), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
    self.bit_depth = bit_depth

  def append(self, observation, action, reward, done):
    if self.symbolic_env:
      self.observations[self.idx] = observation.numpy()
    else:
      self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
    self.actions[self.idx] = action.numpy()
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
    if not self.symbolic_env:
      preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
    return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
    # print(np.asarray([self._sample_idx(L) for _ in range(n)]))
    # [1578 1579 1580 ... 1625 1626 1627]                                                                                                                                        | 0/100 [00:00<?, ?it/s]
    # [1049 1050 1051 ... 1096 1097 1098]
    # [1236 1237 1238 ... 1283 1284 1285]
    # ...
    # [2199 2200 2201 ... 2246 2247 2248]
    # [ 686  687  688 ...  733  734  735]
    # [1377 1378 1379 ... 1424 1425 1426]]
    return [torch.as_tensor(item).to(device=self.device) for item in batch]


class EpisodePrioritizedExperienceReplay():
  def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device, episode_length=None):
    self.device = device
    self.symbolic_env = symbolic_env
    self.size = size
    self.observations = np.empty((size, observation_size) if symbolic_env else (size, 3, 64, 64), dtype=np.float32 if symbolic_env else np.uint8)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32)
    self.nonterminals = np.empty((size, 1), dtype=np.float32)

    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
    self.bit_depth = bit_depth

    self.episode_length = episode_length
    if self.episode_length:
      self.tree = EpisodeSumTree(size, ep_length=self.episode_length)
    else:
      self.tree = SumTree(size)
    self.max_priority = 200

  def append(self, observation, action, reward, done):
    if self.symbolic_env:
      self.observations[self.idx] = observation.numpy()
    else:
      self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
    self.actions[self.idx] = action.numpy()
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

    self.tree.add(self.max_priority, None)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
    if not self.symbolic_env:
      preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
    return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    sample_indices, sample_tree_indices = self._sample_indices(n, L)
    batch = self._retrieve_batch(np.asarray(sample_indices), n, L)
    return [torch.as_tensor(item).to(device=self.device) for item in batch] +[np.array(sample_tree_indices).transpose()]

  def update_priorities(self, indices, priorities):
    for idx, priority in zip(indices.flatten(),priorities.flatten()):
      self.max_priority = max(self.max_priority, priority)
      self.tree.update(idx, priority)

  def _sample_indices(self, n, L): # el otro es n , L batch size y length

    segment = self.tree.total() / n

    data_indices_list = []
    tree_indices_list = []
    i=0
    while len(data_indices_list) < n:
      a = segment * i
      b = segment * (i + 1)
      s = random.uniform(a, b)
      if self.episode_length:
        (tree_indices, p, data_indices) = self.tree.get(s)
      else:
        (idx, p, data_index) = self.tree.get(s)
        data_indices = np.arange(data_index, data_index + L) % self.size
        tree_indices = data_indices +self.tree.capacity-1
      valid_idx = not self.idx in data_indices[1:]  # Make sure data does not cross the memory index
      if valid_idx:
        data_indices_list.append(data_indices)
        tree_indices_list.append(tree_indices)
      i= (i+1)%n
    return data_indices_list, tree_indices_list


class EpisodeCropPrioritizedExperienceReplay():
  def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device, episode_length=None, crop = None):
    self.device = device
    self.symbolic_env = symbolic_env
    self.size = size
    self.crop = crop
    obs_mem_size = (size, *observation_size)
    self.observations = np.empty((size, observation_size) if symbolic_env else obs_mem_size, dtype=np.float32 if symbolic_env else np.uint8)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32)
    self.nonterminals = np.empty((size, 1), dtype=np.float32)

    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
    self.bit_depth = bit_depth

    self.episode_length = episode_length
    if self.episode_length:
      self.tree = EpisodeSumTree(size, ep_length=self.episode_length)
    else:
      self.tree = SumTree(size)
    self.max_priority = 200

  def append(self, observation, action, reward, done):
    if self.symbolic_env:
      self.observations[self.idx] = observation.numpy()
    else:
      self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
    self.actions[self.idx] = action.numpy()
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)
    self.tree.add(self.max_priority, None)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
    return idxs


  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = self.observations[vec_idxs]
    observations = observations.reshape(L, n, *observations.shape[1:])
    key_obs = np.array(observations)
    if self.crop:
      observations = random_crop(observations, self.crop)
      key_obs = random_crop(key_obs, self.crop)
    #To tensor and return
    observations = torch.as_tensor(observations.astype(np.float32))
    key_obs = torch.as_tensor(key_obs.astype(np.float32))
    if not self.symbolic_env:
      preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
      preprocess_observation_(key_obs, self.bit_depth)
    return observations, self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1), key_obs

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    sample_indices, sample_tree_indices = self._sample_indices(n, L)
    batch = self._retrieve_batch(np.asarray(sample_indices), n, L)
    return [torch.as_tensor(item).contiguous().to(device=self.device) for item in batch] +[np.array(sample_tree_indices).transpose()]

  def update_priorities(self, indices, priorities):
    for idx, priority in zip(indices.flatten(),priorities.flatten()):
      self.max_priority = max(self.max_priority, priority)
      self.tree.update(idx, priority)

  def _sample_indices(self, n, L): # el otro es n , L batch size y length

    segment = self.tree.total() / n

    data_indices_list = []
    tree_indices_list = []
    i=0
    while len(data_indices_list) < n:
      a = segment * i
      b = segment * (i + 1)
      s = random.uniform(a, b)
      if self.episode_length:
        (tree_indices, p, data_indices) = self.tree.get(s)
      else:
        (idx, p, data_index) = self.tree.get(s)
        data_indices = np.arange(data_index, data_index + L) % self.size
        tree_indices = data_indices +self.tree.capacity-1
      valid_idx = not self.idx in data_indices[1:]  # Make sure data does not cross the memory index
      if valid_idx:
        data_indices_list.append(data_indices)
        tree_indices_list.append(tree_indices)
      i= (i+1)%n
    return data_indices_list, tree_indices_list