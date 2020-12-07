import os
import cv2
import numpy as np
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
from torch.nn import functional as F
from typing import Iterable
from torch.nn import Module
import vtrace
from skimage.util.shape import view_as_windows
from skimage.transform import rescale, resize



def write_video(frames, title, path=''):
  frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
  _, H, W, _ = frames.shape
  writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
  for frame in frames:
    writer.write(frame)
  writer.release()

def imagine_ahead(prev_state, prev_belief, policy, transition_model, planning_horizon=12):
  '''
  imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
  Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200]) 
  Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
          torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
  '''
  flatten = lambda x: x.view([-1]+list(x.size()[2:]))
  prev_belief = flatten(prev_belief)
  prev_state = flatten(prev_state)
  
  # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
  T = planning_horizon
  beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
  action_list = [torch.ones(1)] * (T+1)
  beliefs[0], prior_states[0] = prev_belief, prev_state

  # Loop over time sequence
  for t in range(T - 1):
    _state = prior_states[t]
    actions = policy.get_action(beliefs[t].detach(),_state.detach())
    action_list[t+1] = actions
    # Compute belief (deterministic hidden state)
    hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
    beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
    # Compute state prior by applying transition dynamics
    hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
    prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
    prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.min_std_dev
    prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])

  #Adding last action for ensemble exp bonus
  actions = policy.get_action(beliefs[t].detach(), _state.detach())
  action_list[T] = actions

  # Return new hidden states
  imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0),
                   torch.stack(prior_std_devs[1:], dim=0), torch.stack(action_list[2:], dim=0)]
  return imagined_traj



def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
  discount_tensor = discount * torch.ones_like(imged_reward) #pcont
  inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
  last = bootstrap
  indices = reversed(range(len(inputs)))
  outputs = []
  for index in indices:
    inp, disc = inputs[index], discount_tensor[index]
    last = inp + disc*lambda_*last
    outputs.append(last)
  outputs = list(reversed(outputs))
  outputs = torch.stack(outputs, 0)
  returns = outputs
  return returns



class ActivateParameters:
  def __init__(self, modules: Iterable[Module]):
      """
      Context manager to locally Activate the gradients.
      example:
      ```
      with ActivateParameters([module]):
          output_tensor = module(input_tensor)
      ```
      :param modules: iterable of modules. used to call .parameters() to freeze gradients.
      """
      self.modules = modules
      self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

  def __enter__(self):
      for param in get_parameters(self.modules):
          # print(param.requires_grad)
          param.requires_grad = True

  def __exit__(self, exc_type, exc_val, exc_tb):
      for i, param in enumerate(get_parameters(self.modules)):
          param.requires_grad = self.param_states[i]

# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
  def __init__(self, modules: Iterable[Module]):
      """
      Context manager to locally freeze gradients.
      In some cases with can speed up computation because gradients aren't calculated for these listed modules.
      example:
      ```
      with FreezeParameters([module]):
          output_tensor = module(input_tensor)
      ```
      :param modules: iterable of modules. used to call .parameters() to freeze gradients.
      """
      self.modules = modules
      self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

  def __enter__(self):
      for param in get_parameters(self.modules):
          param.requires_grad = False

  def __exit__(self, exc_type, exc_val, exc_tb):
      for i, param in enumerate(get_parameters(self.modules)):
          param.requires_grad = self.param_states[i]


# Adapted from https://github.com/rlcode/per/blob/master/SumTree.py

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    def __init__(self, capacity):
        self.write=0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return (idx, self.tree[idx], dataIdx)

class EpisodeSumTree:
    def __init__(self, capacity, ep_length=0):
        self.write=0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.ep_length=ep_length
        self.own_p_tree = np.zeros(2 * capacity - 1) #self priority value
        self.contributed_p_tree = np.zeros(2 * capacity - 1) #contribution from consecutive nodes
        self.pending_idx = set()
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def _smear(self, idx, change):
        low_idx  = np.maximum(self.capacity - 1, idx-self.ep_length)
        if low_idx >= idx:
            return
        self.contributed_p_tree[low_idx:idx]+=change
        for i in np.arange(low_idx,idx):
            self.pending_idx.add(i)
        return

    def total(self):
        self._batch_update()
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.own_p_tree[idx]
        self.own_p_tree[idx] = p
        self._smear(idx, change)
        self.pending_idx.add(idx)
        # self._batch_update()

    def _batch_update(self):
        while self.pending_idx:
            idx = self.pending_idx.pop()
            p_full = self.own_p_tree[idx]+self.contributed_p_tree[idx]
            change_full =  p_full- self.tree[idx]
            self.tree[idx] = p_full
            self._propagate(idx, change_full)

    # get priority and sample sequence starting at s
    def get(self, s):
        self._batch_update()
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        data_indices = np.arange(dataIdx, dataIdx + self.ep_length) % self.capacity
        indices = data_indices +self.capacity-1
        return (indices, self.tree[indices], data_indices)




def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones
    args:
        imgs, batch images with shape (L,B,C,H,W)
    """
    # batch size
    L,B,C,H,W = imgs.shape
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    wl = np.random.randint(0, crop_max, B)
    hl = np.random.randint(0, crop_max, B)

    aux_imgs = imgs.transpose(1, 3, 4, 2, 0)
    windows = view_as_windows(aux_imgs, (1, output_size, output_size, 1, 1))[..., 0, :, :, 0, 0]
    cropped_imgs =windows[np.arange(B), wl, hl].transpose(2, 0, 1, 3, 4)

    return cropped_imgs



def center_crop(img, output_size):
    img_size = img.shape[-1]
    crop_max = img_size - output_size
    l= int(crop_max/2)
    h = l + output_size
    crop_img  = img[..., l:h,l:h]

    return crop_img

def append_dic(dic, key_val_tuples):
    for key, val in key_val_tuples:
        if key in dic.keys():
            dic[key].append(val)
        else:
            dic[key] = [val]
    return dic
