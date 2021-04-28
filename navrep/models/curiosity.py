import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from navrep.tools.curiosity_debug import measure_duration  

class FeatureTransform(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureTransform, self).__init__()
        hidden_size = input_size//2
        self.tf = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x_tf = self.tf(x)
        return x_tf

class ForwardModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ForwardModel, self).__init__()
        # TODO add layers
        # TODO need to decide what input values to consider for the observation (e.g. whole Lidar + robot state + goal or only a seleciton
        # TODO decide upon neural net structure
        hidden_size = input_size//4
        self.fwd = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, action):
        # TODO may need to have another net architeture for ring structure using cnns
        # print(x.shape)
        joint_state = torch.cat((x, action), dim=-1)
        x_next_pred = self.fwd(joint_state)
        return x_next_pred


class ICModule(nn.Module):
    def __init__(self, obsv_size, action_size, use_feature_tf=False, tf_output_size=50):
        super(ICModule, self).__init__()
        self.use_feature_tf = use_feature_tf
        if use_feature_tf :
            self.feature_tf = FeatureTransform(obsv_size, tf_output_size)
            self.dynamics = ForwardModel(tf_output_size+action_size, obsv_size)
        else  :
            self.dynamics = ForwardModel(obsv_size+action_size, obsv_size)

    def forward(self, x_cur, x_next, action):
        if self.use_feature_tf : x_cur = self.feature_tf(x_cur)   
        x_next_est = self.dynamics(x_cur, action)
        rwd = 0.5*torch.mean(torch.square(torch.sub(x_next, x_cur)))
        # TODO add encoder/embedding using inverse model for learning, or use traine VAE model
        return rwd


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


class CuriosityTrainer(object):
    def __init__(self, buffer, model, device, num_iter=200) -> None:
        self.buffer = buffer
        self.model = model
        self.device = device
        self.num_iter = num_iter
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def train_model(self, debug=False):
        running_loss = 0.0
        # TODO revise training structure
        for it in range(self.num_iter):
            # TODO may be replace with random sampled batched from buffer
            #data = self.buffer.get_newest_transition()
            #data = self.buffer.get_complete_buffer()
            data = self.buffer.sample_batch(batch_size=22)

            curr_action = torch.from_numpy(data[0]).float().to(self.device)
            curr_state = torch.from_numpy(
                data[1]).float().squeeze().to(self.device)
            true_next_state = torch.from_numpy(
                data[2]).float().squeeze().to(self.device)

            #data = torch.cat((curr_state, curr_action),0)
            self.optimizer.zero_grad()
            next_state_pred = self.model.dynamics(
                curr_state, curr_action)
            loss = self._loss(next_state_pred, true_next_state)
            loss.backward()
            self.optimizer.step()

            # TODO change
            running_loss = loss.item()
            # print(running_loss)
            if debug and it % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (0, it + 1, running_loss))
                # with torch.no_grad() :
                #    print(np.linalg.norm(next_state_pred.numpy()-true_next_state.numpy(), ord=2)/next_state_pred.shape[0])
                running_loss = 0.0

    def _loss(self, pred_state, true_state):
        return self.criterion(pred_state, true_state)

    def save(self, path):
        raise NotImplementedError()


class CuriosityWrapper(gym.Wrapper):
    def __init__(self, env, use_gpu=True, **kwargs):
        super().__init__(env, **kwargs)
        self.env = env
        self.action_shape = env.action_space.shape
        self.obsv_shape = env.observation_space.shape
        self.cur_reward_scaling = 1
        self.prev_obs = None

        self.buffer_size = 400 #100
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, self.action_shape, self.obsv_shape)

        self.icm = ICModule(
            action_size=self.action_shape[0], obsv_size=self.obsv_shape[0])
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.icm.to(self.device)

        self.trainer = CuriosityTrainer(
            self.replay_buffer, self.icm, self.device)
        self.steps_counter = 0
        self.steps_btw_training = 220

        print(self.icm)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.prev_obs = None
        self.steps_counter = 0
        return self.observation(observation)

    #@measure_duration
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward_out = self.reward(action, reward, observation)

        if self.prev_obs is not None:
            self.replay_buffer.write(self.prev_obs, observation, action)

        if self.steps_counter > 0  and self.steps_counter % self.steps_btw_training == 0:
            print('[{}] Training the intrisic reward module'.format(
                self.steps_counter))
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
            a = torch.from_numpy(action).to(self.device)
            intrinsic_reward = self.icm(s_prev, s_next, a).cpu().numpy()
        else:
            #print('[{}] Starting without intrinsic reward'.format(self.steps_counter))
            intrinsic_reward = 0

        #rew = reward + intrinsic_reward
        rew = intrinsic_reward*100
        # TODO Normalize or not - reward seem to be really

        #print(rew)
        return rew

    def observation(self, observation):
        return observation


if __name__ == '__main__':
    print(torch.cuda.is_available())
    from navrep.envs.e2eenv import *
    import time
    env = E2ENavRepEnv()
    #env = E2E1DNavRepEnv()
    wrapped_env = CuriosityWrapper(env)
    wrapped_env.reset()
    #trainer = CuriosityTrainer(wrapped_env.replay_buffer, wrapped_env.icm)
    num_steps = 1500
    start_time = time.time()
    for i in range(num_steps):
        print('Environment step :  ' + str(i))
        wrapped_env.step(wrapped_env.action_space.sample())
        #wrapped_env.render()
    end_time = time.time()
    total_time = end_time - start_time
    print('Total time :{}, time per step:{}'.format(total_time, total_time/num_steps))


