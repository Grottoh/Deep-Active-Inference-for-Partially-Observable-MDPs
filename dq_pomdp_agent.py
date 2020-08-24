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
        
        self.push_count = 0 # The number of times new data has been pushed (stored) to memory
        
    def push(self, obs, action, reward, done):
        
        # Store data to memory
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

class DQN(nn.Module):

    def __init__(self, n_screens, n_outputs, lr=1e-5, device='cpu'):
        super(DQN, self).__init__()
        
        self.device = device
        
        # The convolutional layers
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
        self.conv3d_size_out = 32*2*8*n_screens
        
        # Two fully connected layers
        self.fc1 = nn.Linear(self.conv3d_size_out, self.conv3d_size_out // 2)
        self.fc2 = nn.Linear(self.conv3d_size_out // 2, n_outputs)
        
        self.optimizer = optim.Adam(self.parameters(), lr)
        
        self.to(self.device)
        
    def forward(self, x):
        # Define the forward pass:
        conv = self.encoder(x)
        h1 = F.relu(self.fc1(conv.view(conv.size(0), -1)))
        y = self.fc2(h1)
        return y
    
class Agent():
    
    def __init__(self, argv):
        
        self.set_parameters(argv) # Set parameters
        
        self.c = 3 # The number of (color) channels of observations
        self.h = 37 # The height of observations (screens)
        self.w = 85 # The width of observations (screens)
        self.obs_shape = (self.c, self.h, self.w)
        
        self.n_actions = self.env.action_space.n # The number of actions available to the agent
        
        self.eps = self.eps_max # Determines how much the agent explores, decreases over time depending on eps_min and eps_decay
        self.freeze_cntr = 0 # Keeps track of when to (un)freeze the target network
        
        # Initialize the policy network and the target network
        self.policy_net = DQN(n_screens=self.n_screens, n_outputs=self.n_actions, lr=self.lr, device=self.device)
        if self.load_network: # If true: load policy network given a path
            self.policy_net.load_state_dict(torch.load(self.network_load_path))
            self.policy_net.eval()
            self.eps = self.eps_min
        self.target_net = DQN(n_screens=self.n_screens, n_outputs=self.n_actions, lr=self.lr, device=self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize the replay memory
        self.memory = ReplayMemory(self.memory_capacity, self.obs_shape, device=self.device)
        
        # Used to pre-process the observations (screens)        
        self.resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
    
        # When sampling from memory at index i, obs_indices indicates that we want observations with indices i-obs_indices, works the same for the others
        self.obs_indices = [self.n_screens-i for i in range(self.n_screens+1)]
        self.action_indices = [1]
        self.reward_indices = [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1
        
    def set_parameters(self, argv):
        
        # The default parameters
        default_parameters = {'run_id':"_rX", 'device':'cuda',
              'env':'CartPole-v1', 'n_episodes':5000, 
              'n_screens':4, 'lr':1e-5,
              'memory_capacity':65536, 'batch_size':32, 'freeze_period':25,
              'gamma':0.99, 'eps_max':0.99, 'eps_min':0.01, 'eps_decay':0.995,
              'print_timer':100,
              'keep_log':True, 'log_path':"logs/dq_pomdp_log{}.txt", 'log_save_timer':10,
              'save_results':True, 'results_path':"results/dq_pomdp_results{}.npz", 'results_save_timer':500,
              'save_network':True, 'network_save_path':"networks/dq_pomdp_policynet{}.pth", 'network_save_timer':500,
              'load_network':False, 'network_load_path':"networks/dq_pomdp_policynet_rX.pth"}
        # Possible command:
            # python dq_pomdp_agent.py device=cuda:0
        
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
        
        self.n_screens = int(custom_parameters['n_screens']) # The number of obervations (screens) that are passed to the policy network
        self.lr = float(custom_parameters['lr']) # The policy network learning rate
        
        self.memory_capacity = int(custom_parameters['memory_capacity']) # The maximum number of items to be stored in memory
        self.batch_size = int(custom_parameters['batch_size']) # The mini-batch size
        self.freeze_period = int(custom_parameters['freeze_period']) # The number of time-steps the target network is frozen
        
        self.gamma = float(custom_parameters['gamma']) # The discount rate
        self.eps_max = float(custom_parameters['eps_max']) # The initial exploration probability
        self.eps_min = float(custom_parameters['eps_min']) # The minimal exploration probability
        self.eps_decay = float(custom_parameters['eps_decay']) # The rate at which the exploration probability decreases
        
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
        if np.random.rand() <= self.eps: # With probability eps take an exploration step
            return torch.randint(low=0, high=self.n_actions, size=(1,)).to(self.device)
        else: # Else let the policy network decide the action
            with torch.no_grad():
                prev_n_obs = self.memory.get_last_n_obs(self.n_screens-1)
                X = torch.cat((prev_n_obs, obs), dim=0).view(1, self.c, self.h, self.w, self.n_screens)
                action_values = self.policy_net(X)
                return torch.tensor([torch.argmax(action_values)], dtype=torch.int64, device=self.device)
        
    def learn(self):
        
        # If there are not enough transitions stored in memory, return
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return
        
        # After every freeze_period time steps, update the target network
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.freeze_cntr += 1
        
        # Retrieve transition data in batches
        all_obs_batch, action_batch, reward_batch, done_batch = self.memory.sample(
                self.obs_indices, self.action_indices, self.reward_indices, self.done_indices, self.max_n_indices, self.batch_size)
        
        # Retrieve a batch of observations for 2 consecutive points in time
        obs_batch = all_obs_batch[:, 0:self.n_screens, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
        next_obs_batch = all_obs_batch[:, 1:self.n_screens+1, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
        
        # Get the q values and the target values, then determine the loss
        value_batch = self.policy_net(obs_batch)
        value_batch = value_batch.gather(1, action_batch)
        target_out = self.target_net(next_obs_batch)
        target_batch = reward_batch + (1-done_batch) * self.gamma * target_out.max(1)[0].view(self.batch_size, 1)
        loss = F.mse_loss(target_batch, value_batch)
        
        self.policy_net.optimizer.zero_grad() # Reset the gradient
        loss.backward() # Compute the gradient
        self.policy_net.optimizer.step() # Perform gradient descent
        
        # Update epsilon
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
            
    def train(self):
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
                msg = "Episodes: {:4d}, eps={:3f}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, self.eps, avg_reward, self.print_timer, last_x)
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
                torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_networks{}_{:d}.pth".format(self.run_id, ith_episode))
        
        self.env.close()
        
        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
            np.savez(self.results_path, np.array(results))
        if self.save_network:
            torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_networks{}_end.pth".format(self.run_id))
            torch.save(self.policy_net.state_dict(), self.network_save_path)
        
        # Print and keep a (.txt) record of stuff
        msg = "Training finished at {}".format(datetime.datetime.now())
        print(msg)
        if self.keep_log:
            self.record.write(msg)
            self.record.close()
                
if __name__ == "__main__":
    agent = Agent(sys.argv[1:])
    agent.train()