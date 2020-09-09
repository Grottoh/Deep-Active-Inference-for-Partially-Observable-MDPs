__author__ = "Otto van der Himst"
__credits__ = "Otto van der Himst, Pablo Lanillos"
__version__ = "1.0"
__email__ = "o.vanderhimst@student.ru.nl"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import sys
import gym
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

class ReplayMemory():
    
    def __init__(self, capacity, obs_shape, device='cpu'):
        
        self.device=device
        
        self.capacity = capacity # The maximum number of items to be stored in memory
        
        self.obs_shape = obs_shape # the shape of observations
        
        # Initialize (empty) memory tensors
        self.obs_mem = torch.empty([capacity]+[dim for dim in self.obs_shape], dtype=torch.float32, device=self.device)
        self.action_mem = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.reward_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.done_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        
        self.push_count = 0 # The number of times new data has been pushed to memory
        
    def push(self, obs, action, reward, done):
        
        # Store data in memory
        self.obs_mem[self.position()] = obs 
        self.action_mem[self.position()] = action
        self.reward_mem[self.position()] = reward
        self.done_mem[self.position()] = done
        
        self.push_count += 1
    
    def position(self):
        # Returns the next position (index) to which data is pushed
        return self.push_count % self.capacity
    
    
    def sample(self, obs_indices, action_indices, reward_indices, done_indices, max_n_indices, batch_size):
        # Fine as long as max_n is not greater than the fewest number of time steps an episode can take
        
        # Pick indices at random
        end_indices = np.random.choice(min(self.push_count, self.capacity)-max_n_indices*2, batch_size, replace=False) + max_n_indices
        
        # Correct for sampling near the position where data was last pushed
        for i in range(len(end_indices)):
            if end_indices[i] in range(self.position(), self.position()+max_n_indices):
                end_indices[i] += max_n_indices
        
        # Retrieve the specified indices that come before the end_indices
        obs_batch = self.obs_mem[np.array([index-obs_indices for index in end_indices])]
        action_batch = self.action_mem[np.array([index-action_indices for index in end_indices])]
        reward_batch = self.reward_mem[np.array([index-reward_indices for index in end_indices])]
        done_batch = self.done_mem[np.array([index-done_indices for index in end_indices])]
        
        # Correct for sampling over multiple episodes
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, max_n_indices):
                if self.done_mem[index-j]:
                    for k in range(len(obs_indices)):
                        if obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.obs_mem[0]) 
                    for k in range(len(action_indices)):
                        if action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.action_mem[0]) # Assigning action '0' might not be the best solution, perhaps as assigning at random, or adding an action for this specific case would be better
                    for k in range(len(reward_indices)):
                        if reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(self.reward_mem[0]) # Reward of 0 will probably not make sense for every environment
                    for k in range(len(done_indices)):
                        if done_indices[k] >= j:
                            done_batch[i, k] = torch.zeros_like(self.done_mem[0]) 
                    break
                
        return obs_batch, action_batch, reward_batch, done_batch
        
    def get_last_n_obs(self, n):
        """ Get the last n observations stored in memory (of a single episode) """
        last_n_obs = torch.zeros([n]+[dim for dim in self.obs_shape], device=self.device)
        
        n = min(n, self.push_count)
        for i in range(1, n+1):
            if self.position() >= i:
                if self.done_mem[self.position()-i]:
                    return last_n_obs
                last_n_obs[-i] = self.obs_mem[self.position()-i]
            else:
                if self.done_mem[-i+self.position()]:
                    return last_n_obs
                last_n_obs[-i] = self.obs_mem[-i+self.position()]
        
        return last_n_obs
    
class Model(nn.Module):
    
    def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=1e-3, softmax=False, device='cpu'):
        super(Model, self).__init__()
        
        self.n_inputs = n_inputs # Number of inputs
        self.n_hidden = n_hidden # Number of hidden units
        self.n_outputs = n_outputs # Number of outputs
        self.softmax = softmax # If true apply a softmax function to the output
        
        self.fc1 = nn.Linear(self.n_inputs, self.n_hidden) # Hidden layer
        self.fc2 = nn.Linear(self.n_hidden, self.n_outputs) # Output layer
        
        self.optimizer = optim.Adam(self.parameters(), lr) # Adam optimizer
        
        self.device = device
        self.to(self.device)
    
    def forward(self, x):
        # Define the forward pass:
        h_relu = F.relu(self.fc1(x))
        y = self.fc2(h_relu)
        
        if self.softmax: # If true apply a softmax function to the output
            y = F.softmax(self.fc2(h_relu), dim=-1).clamp(min=1e-9, max=1-1e-9)
        
        return y

class VAE(nn.Module):
    # In part taken from:
    #   https://github.com/pytorch/examples/blob/master/vae/main.py

    def __init__(self, n_screens, n_latent_states, lr=1e-5, device='cpu'):
        super(VAE, self).__init__()
        
        self.device = device
        
        self.n_screens = n_screens
        self.n_latent_states = n_latent_states
        
        # The convolutional encoder
        self.encoder = nn.Sequential(                
                nn.Conv3d(3, 16, (5,5,1), (2,2,1)),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                
                nn.Conv3d(16, 32, (5,5,1), (2,2,1)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                
                nn.Conv3d(32, 32, (5,5,1), (2,2,1)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)   
                ).to(self.device)
        
        # The size of the encoder output
        self.conv3d_shape_out = (32, 2, 8, self.n_screens)
        self.conv3d_size_out = np.prod(self.conv3d_shape_out)
        
        # The convolutional decoder
        self.decoder = nn.Sequential(
                nn.ConvTranspose3d(32, 32, (5,5,1), (2,2,1)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose3d(32, 16, (5,5,1), (2,2,1)),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose3d(16, 3, (5,5,1), (2,2,1)),
                nn.BatchNorm3d(3),
                nn.ReLU(inplace=True),
                
                nn.Sigmoid()
                ).to(self.device)
        
        # Fully connected layers connected to encoder
        self.fc1 = nn.Linear(self.conv3d_size_out, self.conv3d_size_out // 2)
        self.fc2_mu = nn.Linear(self.conv3d_size_out // 2, self.n_latent_states)
        self.fc2_logvar = nn.Linear(self.conv3d_size_out // 2, self.n_latent_states)
        
        # Fully connected layers connected to decoder
        self.fc3 = nn.Linear(self.n_latent_states, self.conv3d_size_out // 2)
        self.fc4 = nn.Linear(self.conv3d_size_out // 2, self.conv3d_size_out)
        
        self.optimizer = optim.Adam(self.parameters(), lr)
        
        self.to(self.device)

    def encode(self, x):
        # Deconstruct input x into a distribution over latent states
        conv = self.encoder(x)
        h1 = F.relu(self.fc1(conv.view(conv.size(0), -1)))
        mu, logvar = self.fc2_mu(h1), self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Apply reparameterization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, batch_size=1):
        # Reconstruct original input x from the (reparameterized) latent states
        h3 = F.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view([batch_size] + [dim for dim in self.conv3d_shape_out])
        y = self.decoder(deconv_input)
        return y

    def forward(self, x, batch_size=1):
        # Deconstruct and then reconstruct input x
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, batch_size)
        return recon, mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, batch=True):
        if batch:
            BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
            BCE = torch.sum(BCE, dim=(1, 2, 3, 4))
            
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        else:
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + KLD

class Agent():
    
    def __init__(self, argv):
        
        self.set_parameters(argv) # Set parameters
        
        self.c = 3 # The number of (color) channels of observations
        self.h = 37 # The height of observations (screens)
        self.w = 85 # The width of observations (screens)
        self.obs_shape = (self.c, self.h, self.w)
        
        self.n_actions = self.env.action_space.n # The number of actions available to the agent
        
        self.freeze_cntr = 0 # Keeps track of when to (un)freeze the target network
        
        # Initialize the networks:
        self.vae = VAE(self.n_screens, self.n_latent_states, lr=self.lr_vae, device=self.device)
        self.transition_net = Model(self.n_latent_states*2+1, self.n_latent_states, self.n_hidden_trans, lr=self.lr_trans, device=self.device)
        self.policy_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device)
        self.value_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
            
        if self.load_network: # If true: load the networks given paths
            self.vae.load_state_dict(torch.load(self.network_load_path.format("vae"), map_location=self.device))
            self.vae.eval()
            self.transition_net.load_state_dict(torch.load(self.network_load_path.format("trans"), map_location=self.device))
            self.transition_net.eval()
            self.policy_net.load_state_dict(torch.load(self.network_load_path.format("pol"), map_location=self.device))
            self.policy_net.eval()
            self.value_net.load_state_dict(torch.load(self.network_load_path.format("val"), map_location=self.device))
            self.value_net.eval()
        
        if self.load_pre_trained_vae: # If true: load a pre-trained VAE
            self.vae.load_state_dict(torch.load(self.pt_vae_load_path, map_location=self.device))
            self.vae.eval()
        
        # Initialize the replay memory
        self.memory = ReplayMemory(self.memory_capacity, self.obs_shape, device=self.device)
        
        # Used to pre-process the observations (screens)        
        self.resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
    
        # When sampling from memory at index i, obs_indices indicates that we want observations with indices i-obs_indices, works the same for the others
        self.obs_indices = [(self.n_screens+1)-i for i in range(self.n_screens+2)]
        self.action_indices = [2, 1]
        self.reward_indices = [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1
        
    def set_parameters(self, argv):
        
        # The default parameters
        default_parameters = {'run_id':"_rX", 'device':'cuda',
              'env':'CartPole-v1', 'n_episodes':5000, 
              'n_screens':4, 'n_latent_states':32, 'lr_vae':1e-5, 'alpha':25000,
              'n_hidden_trans':64, 'lr_trans':1e-3,
              'n_hidden_pol':64, 'lr_pol':1e-3,
              'n_hidden_val':64, 'lr_val':1e-4,
              'memory_capacity':65536, 'batch_size':32, 'freeze_period':25, 
              'Beta':0.99, 'gamma':12.00,
              'print_timer':100,
              'keep_log':True, 'log_path':"logs/ai_pomdp_log{}.txt", 'log_save_timer':10,
              'save_results':True, 'results_path':"results/ai_pomdp_results{}.npz", 'results_save_timer':500,
              'save_network':True, 'network_save_path':"networks/ai_pomdp_{}net{}.pth", 'network_save_timer':500,
              'load_network':False, 'network_load_path':"networks/ai_pomdp_{}net_rX.pth",
              'pre_train_vae':False, 'pt_vae_n_episodes':500, 'pt_vae_plot':False,
              'load_pre_trained_vae':True, 'pt_vae_load_path':"networks/pre_trained_vae/vae_n{}_end.pth"}
        # Possible commands:
            # python ai_pomdp_agent.py device=cuda:0
            # python ai_pomdp_agent.py device=cuda:0 load_pre_trained_vae=False pre_train_vae=True
        
        # Adjust the custom parameters according to the arguments in argv
        custom_parameters = default_parameters.copy()
        custom_parameter_msg = "Custom parameters:\n"
        for arg in argv:
            key, value = arg.split('=')
            if key in custom_parameters:
                custom_parameters[key] = value
                custom_parameter_msg += "  {}={}\n".format(key, value)
            else:
                print("Argument {} is unknown, terminating.".format(arg))
                sys.exit()
        
        def interpret_boolean(param):
            if type(param) == bool:
                return param
            elif param in ['True', '1']:
                return True
            elif param in ['False', '0']:
                return False
            else:
                sys.exit("param '{}' cannot be interpreted as boolean".format(param))
        
        # Set all parameters
        self.run_id = custom_parameters['run_id'] # Is appended to paths to distinguish between runs
        self.device = custom_parameters['device'] # The device used to run the code
        
        self.env = gym.make(custom_parameters['env']) # The environment in which to train
        self.n_episodes = int(custom_parameters['n_episodes']) # The number of episodes for which to train
        
        # Set number of hidden nodes and learning rate for each network
        self.n_screens = int(custom_parameters['n_screens']) # The number of obervations (screens) that are passed to the VAE
        self.n_latent_states = int(custom_parameters['n_latent_states'])
        self.lr_vae = float(custom_parameters['lr_vae'])
        self.alpha = int(custom_parameters['alpha']) # Used to scale down the VAE's loss
        self.n_hidden_trans = int(custom_parameters['n_hidden_trans'])
        self.lr_trans = float(custom_parameters['lr_trans'])
        self.n_hidden_val = int(custom_parameters['n_hidden_val'])
        self.lr_val = float(custom_parameters['lr_val'])
        self.n_hidden_pol = int(custom_parameters['n_hidden_pol'])
        self.lr_pol = float(custom_parameters['lr_pol'])
        
        self.memory_capacity = int(custom_parameters['memory_capacity']) # The maximum number of items to be stored in memory
        self.batch_size = int(custom_parameters['batch_size']) # The mini-batch size
        self.freeze_period = int(custom_parameters['freeze_period']) # The number of time-steps the target network is frozen
        
        self.Beta = float(custom_parameters['Beta']) # The discount rate
        self.gamma = float(custom_parameters['gamma']) # A precision parameter
        
        self.print_timer = int(custom_parameters['print_timer']) # Print progress every print_timer episodes
        
        self.keep_log = interpret_boolean(custom_parameters['keep_log']) # If true keeps a (.txt) log concerning data of this run
        self.log_path = custom_parameters['log_path'].format(self.run_id) # The path to which the log is saved
        self.log_save_timer = int(custom_parameters['log_save_timer']) # The number of episodes after which the log is saved
        
        self.save_results = interpret_boolean(custom_parameters['save_results']) # If true saves the results to an .npz file
        self.results_path = custom_parameters['results_path'].format(self.run_id) # The path to which the results are saved
        self.results_save_timer = int(custom_parameters['results_save_timer']) # The number of episodes after which the results are saved
        
        self.save_network = interpret_boolean(custom_parameters['save_network']) # If true saves the policy network (state_dict) to a .pth file
        self.network_save_path = custom_parameters['network_save_path'].format("{}", self.run_id) # The path to which the network is saved
        self.network_save_timer = int(custom_parameters['network_save_timer']) # The number of episodes after which the network is saved
        
        self.load_network = interpret_boolean(custom_parameters['load_network']) # If true loads a (policy) network (state_dict) instead of initializing a new one
        self.network_load_path = custom_parameters['network_load_path'] # The path from which to laod the network
        
        self.pre_train_vae = interpret_boolean(custom_parameters['pre_train_vae']) # If true pre trains the vae
        self.pt_vae_n_episodes = custom_parameters['pt_vae_n_episodes'] # The amount of episodes for which to pre train the vae
        self.pt_vae_plot = interpret_boolean(custom_parameters['pt_vae_plot']) # If true plots stuff while training the vae
        
        self.load_pre_trained_vae = interpret_boolean(custom_parameters['load_pre_trained_vae']) # If true loads a pre trained vae
        self.pt_vae_load_path = custom_parameters['pt_vae_load_path'].format(self.n_latent_states) # The path from which to load the pre trained vae
        
        msg = "Default parameters:\n"+str(default_parameters)+"\n"+custom_parameter_msg
        print(msg)
        
        if self.keep_log: # If true: write a message to the log
            self.record = open(self.log_path, "a")
            self.record.write("\n\n-----------------------------------------------------------------\n")
            self.record.write("File opened at {}\n".format(datetime.datetime.now()))
            self.record.write(msg+"\n")
        

    def get_screen(self, env, device='cuda', displacement_h=0, displacement_w=0):
        """
        Get a (pre-processed, i.e. cropped, cart-focussed) observation/screen
        For the most part taken from:
            https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        
        def get_cart_location(env, screen_width):
            world_width = env.x_threshold * 2
            scale = screen_width / world_width
            return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
        
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))
        
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = get_cart_location(env, screen_width)+displacement_w
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        screen = self.resize(screen).unsqueeze(0).to(device)
        screen = screen[:, :, 2:-1, 3:-2]
        return screen
        
    def select_action(self, obs):
        with torch.no_grad():
            # Derive a distribution over states state from the last n observations (screens):
            prev_n_obs = self.memory.get_last_n_obs(self.n_screens-1)
            x = torch.cat((prev_n_obs, obs), dim=0).view(1, self.c, self.h, self.w, self.n_screens)
            state_mu, state_logvar = self.vae.encode(x)
            
            # Determine a distribution over actions given the current observation:
            x = torch.cat((state_mu, torch.exp(state_logvar)), dim=1)
            policy = self.policy_net(x)
            return torch.multinomial(policy, 1)
    
    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, done_batch_t2 = self.memory.sample(
                self.obs_indices, self.action_indices, self.reward_indices,
                self.done_indices, self.max_n_indices, self.batch_size)
        
        # Retrieve a batch of observations for 3 consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0:self.n_screens, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
        obs_batch_t1 = all_obs_batch[:, 1:self.n_screens+1, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
        obs_batch_t2 = all_obs_batch[:, 2:self.n_screens+2, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
        
        # Retrieve a batch of distributions over states for 3 consecutive points in time
        state_mu_batch_t0, state_logvar_batch_t0 = self.vae.encode(obs_batch_t0)
        state_mu_batch_t1, state_logvar_batch_t1 = self.vae.encode(obs_batch_t1)
        state_mu_batch_t2, state_logvar_batch_t2 = self.vae.encode(obs_batch_t2)
        
        # Combine the sufficient statistics (mean and variance) into a single vector
        state_batch_t0 = torch.cat((state_mu_batch_t0, torch.exp(state_logvar_batch_t0)), dim=1)
        state_batch_t1 = torch.cat((state_mu_batch_t1, torch.exp(state_logvar_batch_t1)), dim=1)
        state_batch_t2 = torch.cat((state_mu_batch_t2, torch.exp(state_logvar_batch_t2)), dim=1)
        
        # Reparameterize the distribution over states for time t1
        z_batch_t1 = self.vae.reparameterize(state_mu_batch_t1, state_logvar_batch_t1)
        
        # Retrieve the agent's action history for time t0 and time t1
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)
        
        # At time t0 predict the state at time t1:
        X = torch.cat((state_batch_t0.detach(), action_batch_t0.float()), dim=1)
        pred_batch_t0t1 = self.transition_net(X)

        # Determine the prediction error wrt time t0-t1:
        pred_error_batch_t0t1 = torch.mean(F.mse_loss(
                pred_batch_t0t1, state_mu_batch_t1, reduction='none'), dim=1).unsqueeze(1)
        
        return (state_batch_t1, state_batch_t2, action_batch_t1,
                reward_batch_t1, done_batch_t2, pred_error_batch_t0t1,
                obs_batch_t1, state_mu_batch_t1,
                state_logvar_batch_t1, z_batch_t1)
        
    def compute_value_net_loss(self, state_batch_t1, state_batch_t2,
                           action_batch_t1, reward_batch_t1,
                           done_batch_t2, pred_error_batch_t0t1):
    
        with torch.no_grad():
            # Determine the action distribution for time t2:
            policy_batch_t2 = self.policy_net(state_batch_t2)
            
            # Determine the target EFEs for time t2:
            target_EFEs_batch_t2 = self.target_net(state_batch_t2)
            
            # Weigh the target EFEs according to the action distribution:
            weighted_targets = ((1-done_batch_t2) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)
            
            # Determine the batch of bootstrapped estimates of the EFEs:
            EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t0t1 + self.Beta * weighted_targets
        
        # Determine the EFE at time t1 according to the value network:
        EFE_batch_t1 = self.value_net(state_batch_t1).gather(1, action_batch_t1)
        
        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)
        
        return value_net_loss
    
    def compute_VFE(self, vae_loss, state_batch_t1, pred_error_batch_t0t1):
        
        # Determine the action distribution for time t1:
        policy_batch_t1 = self.policy_net(state_batch_t1)
        
        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_net(state_batch_t1)

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)
        
        # Weigh them according to the action distribution:
        energy_term_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).unsqueeze(1)
        
        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).unsqueeze(1)
        
        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = vae_loss + pred_error_batch_t0t1 + (energy_term_batch - entropy_batch)
        VFE = torch.mean(VFE_batch)
        
        return VFE
    
    def learn(self):
        
        # If there are not enough transitions stored in memory, return
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return
        
        # After every freeze_period time steps, update the target network
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
        self.freeze_cntr += 1
        
        # Retrieve mini-batches of data from memory
        (state_batch_t1, state_batch_t2, action_batch_t1,
        reward_batch_t1, done_batch_t2, pred_error_batch_t0t1,
        obs_batch_t1, state_mu_batch_t1,
        state_logvar_batch_t1, z_batch_t1) = self.get_mini_batches()
        
        # Determine the reconstruction loss for time t1
        recon_batch = self.vae.decode(z_batch_t1, self.batch_size)
        vae_loss = self.vae.loss_function(recon_batch, obs_batch_t1, state_mu_batch_t1, state_logvar_batch_t1, batch=True) / self.alpha
        
        # Compute the value network loss:
        value_net_loss = self.compute_value_net_loss(state_batch_t1, state_batch_t2,
                           action_batch_t1, reward_batch_t1,
                           done_batch_t2, pred_error_batch_t0t1)
        
        # Compute the variational free energy:
        VFE = self.compute_VFE(vae_loss, state_batch_t1.detach(), pred_error_batch_t0t1)
        
        # Reset the gradients:
        self.vae.optimizer.zero_grad()
        self.policy_net.optimizer.zero_grad()
        self.transition_net.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()
        
        # Compute the gradients:
        VFE.backward(retain_graph=True)
        value_net_loss.backward()
        
        # Perform gradient descent:
        self.vae.optimizer.step()
        self.policy_net.optimizer.step()
        self.transition_net.optimizer.step()
        self.value_net.optimizer.step()
    
    def train_vae(self):
        """ Train the VAE separately. """
        
        vae_batch_size = 256
        vae_obs_indices = [self.n_screens-i for i in range(self.n_screens)]
        
        losses = []
        for ith_episode in range(self.pt_vae_n_episodes):
            
            self.env.reset()
            obs = self.get_screen(self.env, self.device)
            done = False
            while not done:
                
                action = self.env.action_space.sample()
                self.memory.push(obs, -99, -99, done)
                
                _, _, done, _ = self.env.step(action)
                obs = self.get_screen(self.env, self.device)
                
                if self.memory.push_count > vae_batch_size + self.n_screens*2:
                    obs_batch, _, _, _ = self.memory.sample(vae_obs_indices, [], [], [], len(vae_obs_indices), vae_batch_size)
                    obs_batch = obs_batch.view(vae_batch_size, self.c, self.h, self.w, self.n_screens)
                    
                    recon, mu, logvar = self.vae.forward(obs_batch, vae_batch_size)
                    loss = torch.mean(self.vae.loss_function(recon, obs_batch, mu, logvar))
                    
                    self.vae.optimizer.zero_grad()
                    loss.backward()
                    self.vae.optimizer.step()
                    
                    losses.append(loss)
                    print("episode %4d: vae_loss=%5.2f"%(ith_episode, loss.item()))
                    
                    if done:
                        if ith_episode > 0 and ith_episode % 10 > 0 and self.pt_vae_plot:
                            plt.plot(losses)
                            plt.show()
                            plt.plot(losses[-1000:])
                            plt.show()
                            for i in range(self.n_screens):
                                plt.imshow(obs_batch[0, :, :, :, i].detach().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
                                plt.show()
                                plt.imshow(recon[0, :, :, :, i].detach().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
                                plt.show()
                
                if done:
                    self.memory.push(obs, -99, -99, done)
                    
                    if ith_episode > 0 and ith_episode % 100 == 0:
                        torch.save(self.vae.state_dict(), "networks/pre_trained_vae/vae_n{}_{:d}.pth".format(
                                self.n_latent_states, ith_episode))
            
        self.memory.push_count = 0
        torch.save(self.vae.state_dict(), "networks/pre_trained_vae/vae_n{}_end.pth".format(self.n_latent_states))
            
    def train(self):
        
        if self.pre_train_vae: # If True: pre-train the VAE
            msg = "Environment is: {}\nPre-training vae. Starting at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
            print(msg)
            if self.keep_log:
                self.record.write(msg+"\n")
            self.train_vae()
            
        msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
        print(msg)
        if self.keep_log:
            self.record.write(msg+"\n")
        
        results = []
        for ith_episode in range(self.n_episodes):
            
            total_reward = 0
            self.env.reset()
            obs = self.get_screen(self.env, self.device)
            done = False
            reward = 0
            
            while not done:
                
                action = self.select_action(obs)
                self.memory.push(obs, action, reward, done)
                
                _, reward, done, _ = self.env.step(action[0].item())
                obs = self.get_screen(self.env, self.device)
                total_reward += reward
                
                self.learn()
                
                if done:
                    self.memory.push(obs, -99, -99, done)
            results.append(total_reward)
            
            # Print and keep a (.txt) record of stuff
            if ith_episode > 0 and ith_episode % self.print_timer == 0:
                avg_reward = np.mean(results)
                last_x = np.mean(results[-self.print_timer:])
                msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x)
                print(msg)
                
                if self.keep_log:
                    self.record.write(msg+"\n")
                    
                    if ith_episode % self.log_save_timer == 0:
                        self.record.close()
                        self.record = open(self.log_path, "a")
            
            # If enabled, save the results and the network (state_dict)
            if self.save_results and ith_episode > 0 and ith_episode % self.results_save_timer == 0:
                np.savez("results/intermediary/intermediary_results{}_{:d}".format(self.run_id, ith_episode), np.array(results))
            if self.save_network and ith_episode > 0 and ith_episode % self.network_save_timer == 0:
                torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_networks{}_{:d}.pth".format(self.run_id, ith_episode))
        
        self.env.close()
        
        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
            np.savez(self.results_path, np.array(results))
        if self.save_network:
            torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_networks{}_end.pth".format(self.run_id))
            torch.save(self.value_net.state_dict(), self.network_save_path)
        
        # Print and keep a (.txt) record of stuff
        msg = "Training finished at {}".format(datetime.datetime.now())
        print(msg)
        if self.keep_log:
            self.record.write(msg)
            self.record.close()
                
if __name__ == "__main__":
    agent = Agent(sys.argv[1:])
    agent.train()