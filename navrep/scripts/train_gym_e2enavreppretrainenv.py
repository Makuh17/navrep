from datetime import datetime
import os

from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.gail import ExpertDataset

from navrep.tools.custom_policy import CustomPolicy, ARCH, _C
from navrep.envs.e2eenv import E2ENavRepEnvPretrain
from navrep.tools.sb_eval_callback import NavrepEvalCallback
from navrep.tools.commonargs import parse_common_args
from navrep.tools.expert_policy import FastmarchORCAPolicy, alt_generate_expert_traj


if __name__ == "__main__":
    args, _ = parse_common_args()

    DIR = os.path.expanduser("~/navrep/models/gym")
    LOGDIR = os.path.expanduser("~/navrep/logs/gym")
    EXPERTDIR = os.path.expanduser("~/navrep/expert_dataset")
    if args.dry_run:
        DIR = "/tmp/navrep/models/gym"
        LOGDIR = "/tmp/navrep/logs/gym"
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    CONTROLLER_ARCH = "_{}_C{}".format(ARCH, _C)
    LOGNAME = "e2enavreppretrainenv_" + START_TIME + "_PPO" + "_E2E" + CONTROLLER_ARCH
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    MODELPATH2 = os.path.join(DIR, "e2enavreppretrainenv_latest_PPO_ckpt")
    EXPERTPATH = os.path.join(EXPERTDIR, "fmORCA")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(EXPERTDIR):
        os.makedirs(EXPERTDIR)

    MILLION = 1000000
    TRAIN_STEPS = args.n
    if TRAIN_STEPS is None:
        TRAIN_STEPS = 60 * MILLION

    N_ENVS = 6
    if args.debug:
        env = DummyVecEnv([lambda: E2ENavRepEnvPretrain(silent=True, scenario='train')]*N_ENVS)
    else:
        env = SubprocVecEnv([lambda: E2ENavRepEnvPretrain(silent=True, scenario='train')]*N_ENVS,
                            start_method='spawn')
    eval_env = E2ENavRepEnvPretrain(silent=True, scenario='train')

    pretrain_env = E2ENavRepEnvPretrain(silent=True, scenario='train')

    def test_env_fn():  # noqa
        return E2ENavRepEnvPretrain(silent=True, scenario='test')
    cb = NavrepEvalCallback(eval_env, test_env_fn=test_env_fn,
                            logpath=LOGPATH, savepath=MODELPATH, verbose=1, render=args.render)

    if not os.path.exists(EXPERTPATH):
        print("Generate expert dataset")
        alt_generate_expert_traj(pretrain_env,500,policy=FastmarchORCAPolicy(), save_path = EXPERTPATH, render=False)

        print("Saved expert data to " + EXPERTPATH)

    dataset = ExpertDataset(expert_path=EXPERTPATH+".npz",traj_limitation=1, batch_size=64)

    model = PPO2(CustomPolicy, env, verbose=1)

    model.pretrain(dataset, n_epochs=1000)

    model.learn(total_timesteps=TRAIN_STEPS+1, callback=cb)
    obs = env.reset()

    model.save(MODELPATH)
    model.save(MODELPATH2)
    print("Model '{}' saved".format(MODELPATH))

    del model

    model = PPO2.load(MODELPATH)

    env = E2ENavRepEnvPretrain(silent=True, scenario='train')
    obs = env.reset()
    for i in range(512):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            env.reset()
#         env.render()
