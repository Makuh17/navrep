import os

from navrep.tools.envplayer import EnvPlayer
#from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.envs.curriculumenv import CurriculumEnv

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    env = CurriculumEnv()
    player = EnvPlayer(env, step_by_step=True)
