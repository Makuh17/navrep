import gym
import numpy as np
import torch
import torch.nn as nn
#from torch.nn.modules.activation import ReLU
import torch.optim as optim

from navrep.tools.curiosity_debug import measure_duration  

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from navrep.envs.encodedenv import EnvEncoder
#import tensorflow as tf
from navrep.models.gpt_curiosity import GPT, GPTConfig, load_checkpoint
import os
from navrep.scripts.train_gpt import _Z, _H
from navrep.tools.rings import generate_rings
from navrep.tools.wdataset import scans_to_lidar_obs
import pathlib

from navrep.envs.e2eenv import RingsLidarAndStateEncoder

_RS = 5
_L = 1080 
NO_VAE_VAR = True
BLOCK_SIZE = 32  # sequence length (context)


class CustomEncoder(object):
    def __init__(self, icm, encoding) -> None:
        super().__init__()
        self.icm = icm

        self.encoding = encoding
        self.rings_def = generate_rings(64, 64)
        self.lidar_mode =  "rings"
        if self.encoding == "V_ONLY":
            self.encoding_dim = _Z + _RS
        elif self.encoding == "VM":
            self.encoding_dim = _Z + _H + _RS
        elif self.encoding == "M_ONLY":
            self.encoding_dim = _H + _RS
        else:
            raise NotImplementedError
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(self.encoding_dim,), dtype=np.float32)

        self._Z = _Z
        self._H = _H
    
    
    def _encode_obs(self, obs, action):
        # convert lidar scan to obs
        lidar_scan = obs[0]  # latest scan only obs (buffer, ray, channel)
        lidar_scan = lidar_scan.reshape(1, _L).astype(np.float32)
        lidar_mode = "rings"
        lidar_obs = scans_to_lidar_obs(lidar_scan, lidar_mode, self.rings_def, channel_first=False)
        self.last_lidar_obs = lidar_obs  # for rendering purposes

        # obs to z, mu, logvar
        
        with torch.no_grad():
            mu, logvar = self.icm.model.encode_mu_logvar(lidar_obs)
        mu = mu[0]
        logvar = logvar[0]
        s = logvar.shape
        if NO_VAE_VAR:
            lidar_z = mu * 1.
        else:
            lidar_z = mu + np.exp(logvar / 2.0) * np.random.randn(*s)

        
        # encode obs through V + M
        self.lidar_z = lidar_z
        if self.encoding == "V_ONLY":
            encoded_obs = np.concatenate([self.lidar_z, obs[1]], axis=0)
        elif self.encoding in ["VM", "M_ONLY"]:
            # get h
            self.gpt_sequence.append(dict(obs=lidar_obs[0], state=obs[1][:2], action=action))
            self.gpt_sequence = self.gpt_sequence[:BLOCK_SIZE]

            with torch.no_grad():
                h = self.icm.model.get_h(self.gpt_sequence)
            
            # encoded obs
            if self.encoding == "VM":
                encoded_obs = np.concatenate([self.lidar_z, obs[1], h], axis=0)
            elif self.encoding == "M_ONLY":
                encoded_obs = np.concatenate([obs[1], h], axis=0)

        return encoded_obs

    def reset(self):
        if self.encoding in ["VM", "M_ONLY"]:
             self.gpt_sequence = []
        self.lidar_z = np.zeros(self._Z)




class ICModule(object):
    def __init__(self, gpu=False, use_img_pred=True):
        super(ICModule, self).__init__()
        gpt_model_path = os.path.expanduser("~/navrep/models/W/navreptraingpt")
        BLOCK_SIZE = 32  # sequence length (context)
        mconf = GPTConfig(BLOCK_SIZE, _H)
        self.model = GPT(mconf, gpu=gpu)
        load_checkpoint(self.model, gpt_model_path, gpu=gpu)
        self.rings_def = generate_rings(64, 64)
        self.lidar_mode =  "rings"
        self.loss = nn.MSELoss()
        self.use_img_pred = use_img_pred
    
    def get_encoded_observation(self, obsv):
        lidar, state = obsv
        scan = lidar.reshape(1, _L).astype(np.float32)
        lidar_obs = scans_to_lidar_obs(scan, self.lidar_mode, self.rings_def, channel_first=False)
        z_true = self.model.encode(lidar_obs)
        return z_true, state
    
    def get_intrinsic_reward(self, obsv_cur, obsv_next, done_cur, action):
        lidar_cur, state_cur = obsv_cur
        lidar_next, state_next = obsv_next
        
        scan_cur = lidar_cur.reshape(1, _L).astype(np.float32)
        scan_next = lidar_next.reshape(1, _L).astype(np.float32)

        lidar_obs_cur = scans_to_lidar_obs(scan_cur, self.lidar_mode, self.rings_def, channel_first=False)
        lidar_obs_next = scans_to_lidar_obs(scan_next, self.lidar_mode, self.rings_def, channel_first=False)
        
        with torch.no_grad() :
            # next state prediction in z space
            img = lidar_obs_cur[0]
            img = img.reshape((1,1,1,64,64))
            #img.shape
            img_t = torch.tensor(img, dtype=torch.float)
            img_t = self.model._to_correct_device(img_t)

            action = action.reshape((1,1,-1))
            #action.shape
            action_t = torch.tensor(action, dtype=torch.float)
            action_t =  self.model._to_correct_device(action_t)

            state_new = state_cur[:2] #only use goal ?
            state_new = state_new.reshape((1,1,-1))
            #state_new.shape
            state_t = torch.tensor(state_new, dtype=torch.float)
            state_t = self.model._to_correct_device(state_t)

            dones = np.array([done_cur])
            dones = dones.reshape((1,1,-1))
            dones_t = torch.tensor(dones, dtype=torch.float)
            dones_t = self.model._to_correct_device(dones_t)

            #print(img_t.get_device())          
            img_pred, state_pred, z_pred, _ = self.model.predict(img_t, state_t, action_t, dones_t)
        

        if self.use_img_pred : 
            img_next = lidar_obs_next[0]
            img_next = img.reshape((1,1,1,64,64))           
            reward = 0.5*np.mean(np.square(np.subtract(img_pred.cpu().numpy(), img_next)))
        else :
            z_pred = z_pred.cpu().numpy()
            #true next state in z space
            z_true = self.model.encode(lidar_obs_next)        
            # prediction error
            reward = 0.5*np.mean(np.square(np.subtract(z_pred, z_true)))
        return reward
       
    



class ICModuleEncodeEnv(object):
    def __init__(self):
        super(ICModuleEncodeEnv, self).__init__()
    def forward(self, x_cur, x_next, action):
        phi_next_est = x_cur[-32:] # TODO check if need to use x_cur  or x_next, problem 512
        return x_next[:32], phi_next_est


# 
class CuriosityWrapper(gym.Wrapper):
    def __init__(self, env, use_gpu=True, tboard_debug=True, obsv_encoding='rings2d', reward_norm=False, feature_tf=None,**kwargs):
        super().__init__(env, **kwargs)
        self.env = env
        self._get_dt = env._get_dt
        self._get_viewer = env._get_viewer

        # TODO implement adaptive
        if reward_norm :
            print('Warning :Only approximate reward normalization implemented')
            self.avg_reward = 0.195343
            self.std_reward = 0.035156
        else :
            self.avg_reward = 0
            self.std_reward = 1

        self.prev_obs = None
        self.feature_tf =  feature_tf
        self.mode = 'GPT_ICM'
       
        self.running_reward = 0
        self.running_extrinsic_rew = 0
        self.tboard_debug = tboard_debug

        self.icm = ICModule(use_gpu)


        # Set encoding of observation used for the RL agent
        if obsv_encoding=='rings2d' :
            self.encoder = RingsLidarAndStateEncoder()
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.observation_space = self.encoder.observation_space
        elif obsv_encoding=='V_ONLY':
            self.encoder = CustomEncoder(self.icm, 'V_ONLY')
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.observation_space = self.encoder.observation_space
        elif obsv_encoding=='VM':
            self.encoder = CustomEncoder(self.icm, 'VM')
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.observation_space = self.encoder.observation_space
        else :
            raise NotImplementedError()


        
        #if icm_module_path is not None :
        #    print('Loading ICM module from : ' + str(icm_module_path))
        #    self.icm.load_state_dict(torch.load(icm_module_path))        
        
        #if use_gpu and torch.cuda.is_available():
        #    self.device = torch.device('cuda:0')
        #else :
        #    self.device = torch.device('cpu')
        #self.icm.to(self.device)        

        #print(self.icm)
        self.done = False
        self.steps_counter = 0
        #debug - tensorboard
        if tboard_debug == True :
            now = datetime.now()
            datestr = now.strftime('%m%d%H%M%S')
            modelparamstr = obsv_encoding
            self.writer = SummaryWriter(log_dir='/home/robin/navrep/tboard/CUR2_{a}_{b}_{c}'.format(a=modelparamstr,b=self.mode,c=datestr))
            self.writer.flush()            
        else :
            self.writer = None
        
        
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if self.done:
            self.prev_obs = None
        self.encoder.reset()
        return self.observation(observation)

    #@measure_duration
    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation

        observation, reward, done, info = self.env.step(action)
        reward_out = self.reward(action, reward, observation, done)
        #print("curiosity reward :" +str(reward_out))
        self.done = done
        self.prev_obs = observation
        self.steps_counter += 1

        return self.observation(observation), reward_out, done, info

    
    def reward(self, action, reward, observation, done):
        if self.prev_obs is not None:
            intrinsic_reward = self.icm.get_intrinsic_reward(self.prev_obs, observation, self.done, action)
        else:
            #print('[{}] Start without intrinsic reward'.format(self.steps_counter))
            intrinsic_reward = 0
        
        #Basic Normalization
        rew = (intrinsic_reward - self.avg_reward)/self.std_reward # + reward/100
                
        # Statistic update
        self.running_reward += rew
        self.running_extrinsic_rew += reward
        if self.tboard_debug and self.steps_counter%10==0:
            self.writer.add_scalar('Curiosity/curiosity_reward_signal', self.running_reward/10 ,self.steps_counter)
            self.writer.add_scalar('Curiosity/extrinsic_reward_signal', self.running_extrinsic_rew/10 ,self.steps_counter)
            self.writer.flush()
            self.running_reward = 0
            self.running_extrinsic_rew = 0
        #print(rew)
        return rew


    def observation(self, observation, action=np.array([0,0,0])):
        return self.encoder._encode_obs(observation,action)




if __name__ == '__main__':
    print(torch.cuda.is_available())
    from navrep.envs.e2eenv import *
    import time
    #env = E2ENavRepEnv()
    #env = E2ENavRepEnvCuriosity()
    #env = E2E1DNavRepEnv()
    #from navrep.envs.navreptrainencodedenv import *
    #env = NavRepTrainEncodedEnvCuriosity(backend='VAE_LSTM', encoding='VM', scenario='train')

    env = NavRepTrainEnv()

    wrapped_env = CuriosityWrapper(env, use_gpu=False, feature_tf=None,tboard_debug=False, obsv_encoding='VM', reward_norm=False)
    #wrapped_env = CuriosityWrapper(env, icm_module_path='/home/robin/navrep/models/curiosity/CURIOSITY_ICM_51_05_12_18_10_18.pt')
    wrapped_env.reset()
    #trainer = CuriosityTrainer(wrapped_env.replay_buffer, wrapped_env.icm)
    num_steps = 1000
    start_time = time.time()

    print(wrapped_env.observation_space.shape)
    average_intrinsic_reward = 0

    for i in range(num_steps):
        #print('Environment step :  ' + str(i))
        obsv, reward, done, info = wrapped_env.step(wrapped_env.action_space.sample())
        #obsv, reward, done, info = wrapped_env.step(np.array([1,0]))
        #print(obsv.shape)
        average_intrinsic_reward += reward
        if done :
            wrapped_env.reset()
            continue
        #wrapped_env.render()
    end_time = time.time()
    total_time = end_time - start_time
    print('Total time :{}, time per step:{}'.format(total_time, total_time/num_steps))
    average_intrinsic_reward = average_intrinsic_reward/num_steps
    print('Average intrinsic reward : {}'.format(average_intrinsic_reward))
    std_dev = 0
    for i in range(num_steps):
        #print('Environment step :  ' + str(i))
        obsv, reward, done, info = wrapped_env.step(wrapped_env.action_space.sample())
        #obsv, reward, done, info = wrapped_env.step(np.array([1,0]))
        
        std_dev += (reward-average_intrinsic_reward)**2
        if done :
            wrapped_env.reset()
            continue
    import math
    std_dev = math.sqrt(std_dev/num_steps)
    print('Std dev intrinsic reward : {}'.format(std_dev))



