import argparse
import os, pickle
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from reaper.env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from reaper.memory import EpisodeCropPrioritizedExperienceReplay as ExperienceReplay
from reaper.utils import write_video, center_crop, append_dic
from tensorboardX import SummaryWriter
from reaper.agents import Dreamer


# Hyperparameters
parser = argparse.ArgumentParser(description='ReaPER arguments')
parser.add_argument('--id', type=str, default='', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='cheetah-run', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--cnn-activation-function', type=str, default='relu', choices=dir(F), help='Model activation function for a convolution layer')
parser.add_argument('--dense-activation-function', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=40, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model-learning-rate', type=float, default=6e-4, metavar='α', help='Learning rate')
parser.add_argument('--actor-learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--value-learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value') 
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')


parser.add_argument('--test-interval', type=int, default=100, metavar='I', help='number of data collection episodes before testing on test environment')
parser.add_argument('--test-episodes', type=int, default=5, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=250, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='if true, save experience replays')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
parser.add_argument('--observation-loss-scale', type=float, default=1, metavar='Bs', help='observation loss scaling factor')
parser.add_argument('--result-dir',  type=str, default='results', help='result folder')

#use RGB inputs or RGBGr?
parser.add_argument('--n_input_channels',  type=int, default=3, help='number of input channels, 3 for RGB, 4 for RGBGray')

#multitask placeholder
parser.add_argument('--n_tasks',  type=int, default=1, help='number of tasks')

#specilaty model training options
#Prioritized Episodic Replay
parser.add_argument('--use-per', type=int, default=0, metavar='H', help='use prioritized experience replay, {0: default priority, 1: sample priority, 2: episodic priority}')

#Contrastive options
parser.add_argument('--use-contrast-loss', type=int, default=0,  help='Use contrastive loss in image embedding')
parser.add_argument('--contrast-loss-scale', type=float, default=1e-1, metavar='Bs', help='contrast loss scaling factor')
parser.add_argument('--crop-size', type=int, default=64, metavar='H', help='crop size')
parser.add_argument('--render-size', type=int, default=80, metavar='H', help='render size')

#State L1 penalty
parser.add_argument('--belief-l1-penalty',type=float, default=0.0, help='l1 belief')


# Collection options for test runs
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--store-test-buffer', action='store_true', help='Create and store experience buffer from test runs')
parser.add_argument('--test-explore', action='store_true', help='add exploration noise on test run policy')
parser.add_argument('--test-metrics', type=int, default=0, help='collect metrics on test run?')
parser.add_argument('--test-collect-interval', type=int, default=1000, help='Collection frequency of test metrics on train runs')



args = parser.parse_args()


print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

# Experiment string
if not args.test:
  id_string = args.id
  if args.use_contrast_loss:
    id_string += '_contrastive_crop_{:d}_render_{:d}_scale_{:.1e}'.format(args.crop_size, args.render_size, args.contrast_loss_scale )
  if args.use_per:
    id_string += '_priority_{:d}_alpha_{:.1f}'.format(args.use_per, args.priority_alpha)
  if args.belief_l1_penalty>0:
    id_string += '_state_l1_{:.1e}'.format(args.belief_l1_penalty)
  if args.observation_loss_scale!= 1.0:
    id_string += '_o_{:.0e}'.format(args.observation_loss_scale)
  if args.action_repeat !=-1:
    id_string += '_ar_{}'.format(args.action_repeat)
  if args.seed != 1:
    id_string += '_seed_{}'.format(args.seed)

  args.id = id_string

# Setup
results_dir = os.path.join('{}'.format(args.result_dir), '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  print("using CUDA")
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  print("using CPU")
  args.device = torch.device('cpu')

# Monitoring metrics
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_rewards': []}



# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, render_size=args.render_size, use_rgbgr=args.n_input_channels==4)
args.action_repeat=env.action_repeat
args.env_action_size = env.action_size


if args.experience_replay is not '' and os.path.exists(args.experience_replay):
  with open(args.experience_replay, 'rb') as pickle_file_handle: #Using pickle to avoid memory error in torch save
    D = pickle.load(pickle_file_handle)
  D.device=args.device
  metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
  print('Loaded experience replay')

elif not args.test:

  D = ExperienceReplay(args.episodes * args.max_episode_length // args.action_repeat,
                        symbolic_env=args.symbolic_env,observation_size=env.observation_size,
                        action_size=env.action_size, bit_depth=args.bit_depth, device=args.device,
                        episode_length= None if args.use_per<2 else args.chunk_size,
                        crop=args.crop_size if args.crop_size != args.render_size else None)
  # Initialise dataset D with S random seed episodes

  for s in range(1, args.seed_episodes + 1):
    observation, done, t = env.reset(), False, 0
    while not done:
      action = env.sample_random_action()
      next_observation, reward, done = env.step(action)
      D.append(observation, action, reward, done)
      observation = next_observation
      t += 1
    metrics['steps'].append(t + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
    metrics['episodes'].append(s)

dreamer_agent = Dreamer(config=args)




# Reload models from savepath if available
if args.models is not '' and os.path.exists(args.models):
  dreamer_agent.load(args.models)

if args.test:
  if args.test_metrics:
    test_metrics = {'beliefs': [], 'beliefs_blind': [],
                    'posterior_states': [], 'prior_states': [], 'prior_states_blind': [],
                    'observations': [], 'estimated_observations': [], 'blind_observations': [],
                    'rewards': [], 'estimated_rewards': [], 'blind_rewards': [], 'mean_rewards':[],
                    'actions': [], 'dones': []}
  # Set models to eval mode
  dreamer_agent.eval()

  if args.store_test_buffer: #using episodes instead of test-epsiodes for maximum length...
    D = ExperienceReplay(args.episodes*args.max_episode_length// args.action_repeat, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth,
                     args.device, crop=args.crop_size if args.crop_size != args.render_size else None, episode_length= None if args.use_per<2 else args.chunk_size,)
  with torch.no_grad():
    total_reward = 0
    for _ in tqdm(range(args.test_episodes)):
      observation = env.reset()
      belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        action, belief, posterior_state, prior_state = dreamer_agent.step(belief, posterior_state, action, observation, env_idx=0, explore=False)
        next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())
        #######################################
        if args.test_metrics:
          estimated_reward = dreamer_agent.world_model.reward_models[0](belief, posterior_state).cpu()
          estimated_observations = dreamer_agent.world_model.observation_model(belief, posterior_state).cpu()
          test_metrics['beliefs'].append(belief.cpu().numpy())
          test_metrics['posterior_states'].append(posterior_state.cpu().numpy())
          test_metrics['prior_states'].append(prior_state.cpu().numpy())
          test_metrics['observations'].append(observation.cpu().numpy())
          test_metrics['estimated_observations'].append(estimated_observations.cpu().numpy())
          test_metrics['rewards'].append(reward)
          test_metrics['estimated_rewards'].append(estimated_reward.cpu().numpy())
          test_metrics['actions'].append(action.cpu().numpy())
          test_metrics['dones'].append(done)
        if args.store_test_buffer:
          D.append(observation, action.cpu(), reward, done)
        total_reward += reward
        observation = next_observation
        if args.render:
          env.render()
        if done:
          pbar.close()
          break

  print('Average Reward:', total_reward / args.test_episodes)
  if args.test_metrics:
    if os.path.exists(args.models):
      fname = args.models.replace('.pth', '_{}_metrics.pkl'.format(args.id))
      with open(fname, 'wb') as pickle_file_handle:
        pickle.dump(test_metrics,pickle_file_handle)
  if args.store_test_buffer:
    fname = args.models.replace('models', 'experience')
    with open(fname, 'wb') as pickle_file_handle:
      pickle.dump(D,pickle_file_handle)
    # torch.save(D, fname)  # Warning: will fail with MemoryError with large memory sizes
    # print(fname)

  env.close()
  quit()

# Training (and testing)
summary_name = results_dir + "/log"
writer = SummaryWriter(summary_name)
# for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=1):
ep_pbar =tqdm(range(metrics['episodes'][-1] + 1, 20*args.episodes + 1), total=20*args.episodes, initial=1)
for episode in ep_pbar:
  # Model fitting
  losses = {}
  print("training loop")

  for s in tqdm(range(args.collect_interval)):
    # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D according to priority from the dataset (including terminal flags)
    observations, actions, rewards, nonterminals, k_observations, sample_indices = D.sample(args.batch_size, args.chunk_size)  # Transitions start at time t = 0
    sample_indices = sample_indices[1:-1]

    model_loss, kl_loss, reward_loss, observation_loss, contrast_loss, actor_loss, value_loss, sample_priority_value = dreamer_agent.compute_loss_and_train(
                                        observations=observations, actions=actions, rewards=rewards,
                                        nonterminals=nonterminals, k_observations=k_observations, env_index=0, backprop=True)

    #Update sample priorities on buffer
    if args.use_per:
      D.update_priorities(sample_indices, sample_priority_value)


    losses = append_dic(losses, [
      ('observation_loss', observation_loss),
      ('reward_loss', reward_loss),
      ('kl_loss', kl_loss),
      ('actor_loss', actor_loss),
      ('value_loss',  value_loss),
      ('mean_rewards', rewards.mean().item())])
    if args.use_contrast_loss:
      losses = append_dic(losses, [
        ('contrast_loss', contrast_loss)])

  # Update metrics
  metric_kv_tuples = [(k,v) for k,v in losses.items()]
  metrics = append_dic(metrics, metric_kv_tuples)

  # Data collection
  print("Data collection")
  with torch.no_grad():
    observation, total_reward = env.reset(), 0
    belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
    pbar = tqdm(range(args.max_episode_length // args.action_repeat))
    exceeds_current_buffer = metrics['steps'][-1]+1>D.steps
    for t in pbar:
      # print("step",t)
      action, belief, posterior_state, prior_state = dreamer_agent.step(belief, posterior_state, action, observation,
                                                                        env_idx=0, explore=True)
      next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())

      if exceeds_current_buffer:
        D.append(observation, action.cpu(), reward, done)
      total_reward += reward
      observation = next_observation
      if args.render:
        env.render()
      if done:
        pbar.close()
        break
    
    # Update and plot train reward metrics
    metrics['steps'].append(t+1 + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)



  # Test model
  print("Test model")
  if episode % args.test_interval == 0:
    # Set models to eval mode
    dreamer_agent.eval()

    # Initialise parallelised test environments
    test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, ), {'render_size':args.render_size, 'use_rgbgr':True}, args.test_episodes)
    test_metrics = {'beliefs': [], 'beliefs_blind':[],
                    'posterior_states': [],'prior_states':[],'prior_states_blind':[],
                    'observations':[], 'estimated_observations':[], 'blind_observations':[],
                    'rewards': [],'estimated_rewards': [],'blind_rewards': [],
                    'actions': [], 'dones': []}
    with torch.no_grad():
      observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes, )), []
      belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size, device=args.device), torch.zeros(args.test_episodes, args.state_size, device=args.device), torch.zeros(args.test_episodes, env.action_size, device=args.device)
      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        action, belief, posterior_state, prior_state = dreamer_agent.step(belief, posterior_state, action, observation, env_idx=0, explore=False)
        next_observation, reward, done = test_envs.step(action.cpu() if isinstance(test_envs, EnvBatcher) else action[0].cpu())

        total_rewards += reward.numpy()
        estimated_reward = dreamer_agent.world_model.reward_models[0](belief, posterior_state).cpu()
        if not args.symbolic_env:  # Collect real vs. predicted frames for video
          estimated_observations =  dreamer_agent.world_model.observation_model(belief, posterior_state).cpu()
          if args.crop_size!= args.render_size:
            video_frames.append(make_grid(torch.cat([center_crop(observation[:,-3:,...], args.crop_size), estimated_observations[:,-3:,...]], dim=3) + 0.5,nrow=3).numpy())  # Decentre
          else:
            video_frames.append(make_grid(torch.cat([observation[:,-3:,...],estimated_observations[:,-3:,...] ], dim=3) + 0.5, nrow=3).numpy())  # Decentre
          if episode % args.test_collect_interval== 0:
            test_metrics['beliefs'].append(belief.cpu().numpy())
            test_metrics['posterior_states'].append(posterior_state.cpu().numpy())
            test_metrics['prior_states'].append(prior_state.cpu().numpy())
            test_metrics['observations'].append(observation.cpu().numpy())
            test_metrics['estimated_observations'].append(estimated_observations.cpu().numpy())
            test_metrics['rewards'].append(reward.cpu().numpy())
            test_metrics['estimated_rewards'].append(estimated_reward.cpu().numpy())
            test_metrics['actions'].append(action.cpu().numpy())
            test_metrics['dones'].append(done.cpu().numpy())
        observation = next_observation
        if done.sum().item() == args.test_episodes:
          pbar.close()
          break

    # Update and plot reward metrics (and write video if applicable) and save metrics
    metrics = append_dic(metrics, [('test_episodes',episode),('test_rewards',total_rewards.tolist())])
    # metrics['test_episodes'].append(episode)
    # metrics['test_rewards'].append(total_rewards.tolist())
    if not args.symbolic_env:
      episode_str = str(episode).zfill(len(str(args.episodes)))
      write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
      save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
      if episode % args.test_collect_interval== 0:
        with open(os.path.join(results_dir, 'test_episode_metrics_%s.pkl' % episode_str), 'wb') as pickle_file_handle:
          pickle.dump(test_metrics, pickle_file_handle)
        del test_metrics
      save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
    torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    writer.add_scalar("test_reward", np.array(metrics['test_rewards']).flatten()[-1], metrics['steps'][-1])
    # Set models to train mode
    dreamer_agent.train()
    # Close test environments
    test_envs.close()

  writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
  writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
  writer.add_scalar("observation_loss", np.array(metrics['observation_loss']).flatten()[-1], metrics['steps'][-1])
  writer.add_scalar("reward_loss", np.array(metrics['reward_loss']).flatten()[-1], metrics['steps'][-1])
  writer.add_scalar("kl_loss",np.array(metrics['kl_loss']).flatten()[-1], metrics['steps'][-1])
  writer.add_scalar("actor_loss", np.array(metrics['actor_loss']).flatten()[-1], metrics['steps'][-1])
  writer.add_scalar("value_loss", np.array(metrics['value_loss']).flatten()[-1], metrics['steps'][-1])
  if args.use_contrast_loss:
    writer.add_scalar("contrast_loss",np.array(metrics['contrast_loss']).flatten()[-1], metrics['steps'][-1])
  # writer.add_scalar("mean_belief_distance",np.array(metrics['belief_distance_mean']).flatten()[-1], metrics['steps'][-1])
  print("episodes: {}, total_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]))

  #Write metrics directly onto pickle
  with open(os.path.join(results_dir, 'metrics.pkl'), 'wb') as pickle_file_handle:
    pickle.dump(metrics,pickle_file_handle)

  # Checkpoint models
  if episode % args.checkpoint_interval == 0 or (episode == args.episodes) or metrics['steps'][-1]>= args.episodes*args.max_episode_length// args.action_repeat  :
    dreamer_agent.save(os.path.join(results_dir, 'models_%d.pth' % episode))
    if args.checkpoint_experience:
      torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes

  if metrics['steps'][-1]>= args.episodes*args.max_episode_length// args.action_repeat:
    ep_pbar.close()
    break



# Close training environment
env.close()