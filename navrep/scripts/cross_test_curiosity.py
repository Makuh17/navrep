from crowd_sim.envs.policy.policy import Policy
from navrep.envs.encodedenv import EncodedEnv
from stable_baselines.common.policies import MlpPolicy
#from navrep.envs.e2eenv import E2ENavRepEnvCuriosity
import os
from stable_baselines import PPO2
#from navrep.models.curiosity import CuriosityWrapper
from navrep.tools.custom_policy import CustomPolicy, Custom1DPolicy
from navrep.scripts.cross_test_navreptrain_in_ianenv import run_test_episodes

from navrep.envs.navreptrainencodedenv import *
from navrep.envs.e2eenv import E2ENavRepEnv
from navrep.envs.e2eenv import E2EIANEnv

class CuriosityPolicy(object):
    """ thin wrapper for gym policies """
    def __init__(self, model_path=None, model=None):
        if model is not None:
            self.model = model
        else:
            self.model_path = model_path
            if self.model_path is None:
                self.model_path = os.path.expanduser(
                    "~/navrep/models/gym/CURIOSITY_V2_e2enavreptrainenv_2021_05_25__10_06_51_PPO_E2E_VCARCH_C64_ckpt.zip")
            #self.model = PPO2.load(self.model_path, policy=CustomPolicy)
            self.model = PPO2.load(self.model_path, policy=MlpPolicy)
            print("Model '{}' loaded".format(self.model_path))

    def act(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action


if __name__=='__main__':
    #from navrep.envs.e2eenv import E2EIANEnv
    #env = E2EIANEnv(silent=True, collect_trajectories=False)
    #env = CuriosityWrapper(E2ENavRepEnv(silent=True, scenario='train'))
    env = EncodedEnv(backend='VAE_LSTM', encoding='VM')
    #env = NavRepTrainEncodedEnv(backend='VAE_LSTM', encoding='VM', silent=True, scenario='train')
    policy = CuriosityPolicy()
    S = run_test_episodes(env, policy, render=True, num_episodes=20)
