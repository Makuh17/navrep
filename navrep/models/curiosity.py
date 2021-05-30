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
import tensorflow as tf
import os
import pathlib


_RS = 5

# TODO 
# Can only deal with the 1D env currently, other would need larger architectural change,
#i.e. would need to have access to the raw lidar scan data
class VaeFeatureTransform(nn.Module):
    def __init__(self, use_gpu=True):
        super(VaeFeatureTransform, self).__init__()    
        # otherwise cuda doesnt work :
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else :
            self.device = torch.device('cpu')  
        self.encoder = EnvEncoder(backend='VAE_LSTM', encoding='V_ONLY',
            vae_model_path=os.path.expanduser("~/navrep/models/V/navreptrainvae.json"))
    def forward(self, x):
        # TODO not really efficient ..
        # TODO doesnt work since require lidar measurements ?
        obs = x.cpu().numpy()
        # TODO Problem cant handle batched
        obs_comb = np.split(obs, [1080],axis=0)
        encoded = self.encoder._encode_obs(obs_comb, action=None)
        encoded = torch.from_numpy(encoded).to(self.device)
        return encoded

class FeatureTransform(nn.Module):
    def __init__(self, output_size=64, use_conv=True, channels_in=1, num_obsv=1):
        super(FeatureTransform, self).__init__()
        self.num_obsv = num_obsv
        self.conv_size = 64*self.num_obsv

        if use_conv :
            self.conv = nn.Sequential(
                nn.Conv2d(channels_in, 32, 4, stride=2),
                #nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32,64,4, stride=2),
                #nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(64, 128, 4, stride=2),
                #nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(128,256,4, stride=2),
                #nn.BatchNorm2d(32),
                nn.ReLU()
            )
            self.dense = nn.Sequential(
                nn.Linear(256*2*2*self.num_obsv,32)
            )
            self.fcn = nn.Sequential(
                nn.Linear(32 + _RS, 64),
                nn.ReLU(),
                nn.Linear(64, output_size),
                nn.ReLU()
            )
        else :
            raise NotImplementedError()

        self._set_grad_zero()

    def _set_grad_zero(self):
        for param in self.dense.parameters():
            param.requires_grad = False
        for param in self.conv.parameters():
            param.requires_grad = False
        for param in self.fcn.parameters():
            param.requires_grad = False

    def forward(self,x):
        #print(x.shape)
        x_lidar = x[:, :-_RS]
        x_state = x[:,-_RS:]

        rep_2d = x_lidar.view((-1, 1, 64, self.conv_size))
        robot_state = x_state.view((-1,5))

        h = self.conv(rep_2d)
        h = h.view((-1, 2*2*self.num_obsv*256))
        h = self.dense(h)

        extracted_features = torch.cat((h,robot_state), dim=1)
        out = self.fcn(extracted_features)

        return out


class RandomFeatureTransform2(nn.Module):
    def __init__(self, output_size=64, num_obsv=1):
        super(RandomFeatureTransform2, self).__init__()
        self.tf = FeatureTransform(output_size=output_size, num_obsv=num_obsv)
        self.tf._set_grad_zero()

    def forward(self, x):
        return self.tf.forward(x)


class RandomFeatureTransform(nn.Module):
    def __init__(self, input_size, output_size):
        super(RandomFeatureTransform, self).__init__()
        hidden_size = input_size//2
        self.tf = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        #TODO maybe change init of weights with nn.init
        for param in self.tf.parameters():
            param.grad = None
    def forward(self, x):
        x_tf = self.tf(x)
        return x_tf



class ForwardModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(ForwardModel, self).__init__()
        # TODO need to decide what input values to consider for the observation (e.g. whole Lidar + robot state + goal or only a seleciton
        # TODO decide upon neural net structure
        # TODO dropout, batch norm and co
        #hidden_size = input_size//2
        self.fwd = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, action):
        # print(x.shape)
        joint_state = torch.cat((x, action), dim=-1)
        x_next_pred = self.fwd(joint_state)
        return x_next_pred



class ForwardModel2(nn.Module):
    def __init__(self, obsv_size, action_size, use_conv=True, channels_in=1, num_obsv=1):
        super(ForwardModel2, self).__init__()
        # num_obsv unused, for future use if want to have input multiple states as inputs (not yet fully implemented)
        self.num_obsv = num_obsv
        self.conv_size = 64*self.num_obsv

        if use_conv :
            self.conv = nn.Sequential(
                nn.Conv2d(channels_in, 32, 4, stride=2),
                #nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32,64,4, stride=2),
                #nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(64, 128, 4, stride=2),
                #nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(128,256,4, stride=2),
                #nn.BatchNorm2d(32),
                nn.ReLU()
            )
            self.dense = nn.Sequential(
                nn.Linear(256*2*2*self.num_obsv,32)
            )
            self.fcn = nn.Sequential(
                nn.Linear(32 + _RS + action_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, obsv_size),
                nn.ReLU()
            )
        else :
            raise NotImplementedError()


    def forward(self,x, action):
        #print(x.shape)
        x_lidar = x[:, :-_RS]
        x_state = x[:,-_RS:]
        #print(x_lidar.shape)
        rep_2d = x_lidar.view((-1, 1, 64, self.conv_size))
        robot_state = x_state.view((-1,5))

        h = self.conv(rep_2d)
        h = h.view((-1, 2*2*self.num_obsv*256))
        h = self.dense(h)

        extracted_features = torch.cat((h,robot_state,action), dim=1)
        out = self.fcn(extracted_features)

        return out




class ICModule(nn.Module):
    def __init__(self, obsv_size, action_size, use_feature_tf=None, tf_output_size=64):
        super(ICModule, self).__init__()
        self.use_feature_tf = use_feature_tf
        self.fwd_dynamics_name='cnn_fwd'
        if use_feature_tf != None:
            if use_feature_tf=='random' :
                #self.feature_tf = RandomFeatureTransform(obsv_size, tf_output_size)
                self.feature_tf = RandomFeatureTransform2(output_size=tf_output_size)
            elif use_feature_tf =='trained':
                self.feature_tf = FeatureTransform(output_size=tf_output_size)
            elif use_feature_tf =='random_encoded':
                self.feature_tf = RandomFeatureTransform(obsv_size, tf_output_size)
            elif use_feature_tf=='vae' :
                raise NotImplementedError()
                #self.feature_tf = VaeFeatureTransform()
                #tf_output_size = 37
            else :
                raise NotImplementedError()
            
            self.dynamics = ForwardModel(tf_output_size+action_size, tf_output_size)
        else  :
            self.dynamics = ForwardModel(obsv_size+action_size, obsv_size)
            #self.dynamics = ForwardModel2(obsv_size, action_size)

    def forward(self, x_cur, x_next, action):
        if self.use_feature_tf :
            x_cur = self.feature_tf(x_cur)
            x_next = self.feature_tf(x_next)

        phi_next_est = self.dynamics(x_cur, action)
        return x_next, phi_next_est
    
    def get_properties_string(self):
        return "{a}_{b}_".format(a=self.use_feature_tf, b=self.fwd_dynamics_name)


class ReplayBuffer(object):
    def __init__(self, size, action_shape, observation_shape) -> None:
        super().__init__()
        self.next_idx = 0
        self.last_idx = 0
        self.size = size
        self.full = False

        self.action_buffer = np.empty([self.size, ] + list(action_shape))
        self.obsv_buffer = np.empty([self.size, ] + list(observation_shape))
        self.next_obsv_buffer = np.empty(
            [self.size, ] + list(observation_shape))

    def write(self, obsv, obsv_next, action):
        self.action_buffer[self.next_idx] = action
        self.obsv_buffer[self.next_idx] = obsv
        self.next_obsv_buffer[self.next_idx] = obsv_next

        self.last_idx = self.next_idx
        self.next_idx += 1
        if self.next_idx == self.size :
            self.full = True
            self.next_idx = 0

    def get_newest_transition(self):
        return self.action_buffer[self.last_idx], self.obsv_buffer[self.last_idx], self.next_obsv_buffer[self.last_idx]

    def get_k_newest_transitions(self,k):
        if self.last_idx-k >= 0 :
            return self.action_buffer[self.last_idx-k:self.last_idx], self.obsv_buffer[self.last_idx-k:self.last_idx], self.next_obsv_buffer[self.last_idx-k:self.last_idx]
        else :
            return self.action_buffer[0:self.last_idx], self.obsv_buffer[0:self.last_idx], self.next_obsv_buffer[0:self.last_idx]
 

    def get_complete_buffer(self):
        if self.full :
            return self.action_buffer, self.obsv_buffer, self.next_obsv_buffer
        else :
            return self.action_buffer[0:self.last_idx], self.obsv_buffer[0:self.last_idx], self.next_obsv_buffer[0:self.last_idx]
        
    def get_newest_batch(self, batch_size):
        raise NotImplementedError()

    def sample_batch(self, batch_size):
        if self.full :
            rand_idxs = np.random.randint(0, self.size, size=batch_size)
        else :
            rand_idxs = np.random.randint(0, self.next_idx, size=batch_size) 
        return self.action_buffer[rand_idxs], self.obsv_buffer[rand_idxs], self.next_obsv_buffer[rand_idxs]
    
    def _save_buffer(self):
        import pathlib
        path =  pathlib.Path.home() / 'navrep/buffer/'
        path.mkdir(parents=True, exist_ok=True)
        print("Saving buffer to " + str(path))
        np.save(path/ 'action' ,self.action_buffer)
        np.save(path / 'obsv', self.obsv_buffer)
        np.save(path / 'obsvnext', self.next_obsv_buffer)

    def _load_buffer(self):
        import pathlib
        path =  pathlib.Path.home() / 'navrep/buffer/'
        path.mkdir(parents=True, exist_ok=True)
        self.action_buffer = np.load(path/ 'action.npy')
        self.obsv_buffer = np.load(path / 'obsv.npy')
        self.next_obsv_buffer = np.load(path / 'obsvnext.npy')
        self.full = True
        new_size = self.action_buffer.shape[0]
        if(new_size!=self.size) :
            print("Size of buffer has changed !")



class CuriosityTrainer(object):
    def __init__(self, buffer, model, device, num_iter=16, tboard_writer=None, verbose=0) -> None:
        self.buffer = buffer
        self.model = model
        self.device = device
        self.num_iter = num_iter
        self.num_training_session = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.tboard_writer = tboard_writer
        self.verbose = verbose

        self.save_model = True 
        now = datetime.now()
        run_name = now.strftime('_%m_%d_%H_%M_%S')
        if self.save_model :              
            self.model_path =  pathlib.Path.home() / 'navrep/models/curiosity' /run_name
            self.model_path.mkdir(parents=True, exist_ok=True)
        self.saving_interval = 50
        

    def train_model(self, debug=True):
        running_loss = 0.0
        # TODO revise training structure
        for it in range(self.num_iter):
            #data = self.buffer.get_newest_transition()
            #data = self.buffer.get_complete_buffer()
            data = self.buffer.sample_batch(batch_size=32)

            cur_action = torch.from_numpy(data[0]).float().to(self.device)
            cur_state = torch.from_numpy(
                data[1]).float().squeeze().to(self.device)
            true_next_state = torch.from_numpy(
                data[2]).float().squeeze().to(self.device)

            self.optimizer.zero_grad()
            phi_next, phi_next_est = self.model(cur_state, true_next_state, cur_action)
            loss = self._loss(phi_next_est, phi_next)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()           
            if debug and it % 9 == 0:
                if self.verbose > 0 :
                    print('[%d, %5d] loss: %.3f' %(0, it + 1, running_loss))

                self.tboard_writer.add_scalar('ForwardModel/training_loss', running_loss/10 ,self.num_iter*self.num_training_session + it)
                #    print(np.linalg.norm(next_state_pred.numpy()-true_next_state.numpy(), ord=2)/next_state_pred.shape[0])
                self.tboard_writer.flush()
                running_loss = 0.0
        if self.save_model and self.num_training_session % self.saving_interval-1 == 0:
            self._save_model()

        self.num_training_session += 1
    def _loss(self, pred_state, true_state):
        return self.criterion(pred_state, true_state)
    
    def _save_model(self):
        now = datetime.now()
        namestr = 'CURIOSITY_ICM_' + self.model.get_properties_string() + str(self.num_training_session) + now.strftime('_%m_%d_%H_%M_%S') +'.pt'
        torch.save(self.model.state_dict(), self.model_path/ namestr)

        

class CuriosityWrapper(gym.Wrapper):
    def __init__(self, env, use_gpu=True, tboard_debug=True, icm_module_path=None, feature_tf='random', verbose_icm_training=0, **kwargs):
        super().__init__(env, **kwargs)
        self.env = env
        #self._get_dt = env._get_dt
        #self._get_viewer = env._get_viewer

        self.action_shape = env.action_space.shape
        self.obsv_shape = env.observation_space.shape

        self.cur_reward_scaling = 1e4
        self.prev_obs = None
        self.feature_tf =  feature_tf
        #self.feature_tf = None
        self.comment = 'done_reward_fixed_'
        self.steps_btw_training = 128#number of rollouts between training of the fwd model 
        self.buffer_size = 1000 # TODO
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, self.action_shape, self.obsv_shape)
        self.running_reward = 0
        self.running_extrinsic_rew = 0
        self.tboard_debug = tboard_debug

        self.icm = ICModule(
            action_size=self.action_shape[0], obsv_size=self.obsv_shape[0],use_feature_tf=self.feature_tf)
        
        if icm_module_path is not None :
            print('Loading ICM module from : ' + str(icm_module_path))
            self.icm.load_state_dict(torch.load(icm_module_path))
        
        
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else :
            self.device = torch.device('cpu')

        self.icm.to(self.device)        

        print(self.icm)
        self.done = False
        #debug - tensorboard
        if tboard_debug == True :
            now = datetime.now()
            datestr = now.strftime('%m%d%H%M%S')
            modelparamstr = '_B:' + str(self.buffer_size) + '_FTF:' +str(self.feature_tf) + '_SBT:' +str(self.steps_btw_training) + '_CRS:' +str(self.cur_reward_scaling)
            self.writer = SummaryWriter(log_dir='/home/robin/navrep/tboard/CUR_{a}_{b}_{c}'.format(a=modelparamstr,b=self.comment,c=datestr))
            self.writer.flush()            
        else :
            self.writer = None

        self.trainer = CuriosityTrainer(
            self.replay_buffer, self.icm, self.device, tboard_writer=self.writer, verbose=verbose_icm_training)
        self.steps_counter = 0
        
        
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if self.done:
            self.prev_obs = None
        return self.observation(observation)

    #@measure_duration
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        #print(reward)
        reward_out = self.reward(action, reward, observation)
        #print("curiosity :" +str(reward_out))

        self.done = done

        if self.prev_obs is not None:
            self.replay_buffer.write(self.prev_obs, observation, action)

        if self.steps_counter > 0  and self.steps_counter % self.steps_btw_training == 0:
            #print('[{}] Training the intrisic reward module'.format(
            #    self.steps_counter))
            self.trainer.train_model()

        self.prev_obs = observation
        self.steps_counter += 1
        return observation, reward_out, done, info

    @torch.no_grad()
    def reward(self, action, reward, observation):
        # TODO may need to wait longer until NN is properly trained, since initial rewards can be really high
        if self.prev_obs is not None:
            s_prev = torch.from_numpy(
                self.prev_obs).float().squeeze().to(self.device)
            s_next = torch.from_numpy(
                observation).float().squeeze().to(self.device)
            s_next = s_next.view([-1] + list(s_next.size()))
            s_prev = s_prev.view([-1] +list(s_prev.size()))
            #print(s_prev.shape)
            a = torch.from_numpy(action).float().to(self.device)
            a = a.view([-1] + list(a.size()))
            phi_next, phi_next_est = self.icm(s_prev, s_next, a)
            intrinsic_reward = 0.5*torch.mean(torch.square(torch.sub(phi_next, phi_next_est))).cpu().numpy()
        else:
            #print('[{}] Starting without intrinsic reward'.format(self.steps_counter))
            intrinsic_reward = 0
        #rew = reward + intrinsic_reward
        rew = intrinsic_reward*self.cur_reward_scaling # + reward/100
        # TODO Normalize or not - reward seem to be really

        # TODO remove
        if rew < 100 :
            rew = 0
        
        self.running_reward += rew
        self.running_extrinsic_rew += reward
        if self.tboard_debug and self.steps_counter%10==0:
            self.writer.add_scalar('Curiosity/reward_signal', self.running_reward/10 ,self.steps_counter)
            self.writer.add_scalar('Curiosity/extrinsic_reward_signal', self.running_extrinsic_rew/10 ,self.steps_counter)
            self.writer.flush()
            self.running_reward = 0
            self.running_extrinsic_rew = 0
        #print(rew)
        return rew
    def observation(self, observation):
        return observation


if __name__ == '__main__':
    print(torch.cuda.is_available())
    from navrep.envs.e2eenv import *
    import time
    #env = E2ENavRepEnv()
    #env = E2ENavRepEnvCuriosity()
    #env = E2E1DNavRepEnv()
    from navrep.envs.navreptrainencodedenv import *
    env = NavRepTrainEncodedEnvCuriosity(backend='VAE_LSTM', encoding='V_ONLY', scenario='train')

    wrapped_env = CuriosityWrapper(env, use_gpu=False, feature_tf=None,verbose_icm_training=1)
    #wrapped_env = CuriosityWrapper(env, icm_module_path='/home/robin/navrep/models/curiosity/CURIOSITY_ICM_51_05_12_18_10_18.pt')
    wrapped_env.reset()
    #trainer = CuriosityTrainer(wrapped_env.replay_buffer, wrapped_env.icm)
    num_steps = 1000
    start_time = time.time()

    print(wrapped_env.observation_space.shape)

    for i in range(num_steps):
        print('Environment step :  ' + str(i))
        #obsv, reward, done, info = wrapped_env.step(wrapped_env.action_space.sample())
        obsv, reward, done, info = wrapped_env.step(np.array([1,0]))
        print(obsv.shape)
        if done :
            wrapped_env.reset()
            continue
        wrapped_env.render()
    end_time = time.time()
    total_time = end_time - start_time
    print('Total time :{}, time per step:{}'.format(total_time, total_time/num_steps))


