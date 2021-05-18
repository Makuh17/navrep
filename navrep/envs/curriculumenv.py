from navrep.envs.navreptrainenv import NavRepTrainEnv, SDOADRLDummyPolicy

import configparser
import time
import gym
from gym import spaces
import numpy as np
from pandas import DataFrame
import tensorflow as tf
import threading
import os
from CMap2D import flatten_contours, render_contours_in_lidar, CMap2D, CSimAgent, fast_2f_norm
from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose
from pkg_resources import resource_filename

import crowd_sim  # adds CrowdSim-v0 to gym  # noqa
from crowd_sim.envs.crowd_sim import CrowdSim  # reference to env code  # noqa
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.network_om import SDOADRL
from crowd_sim.envs.utils.action import ActionXYRot
from crowd_sim.envs.utils.info import Collision, CollisionOtherAgent, ReachGoal

PROGRESS_WEIGHT = 0.001


# Crowd sim dependencies
from numpy.linalg import norm
import collections
Obstacle = collections.namedtuple(
    "Obstacle", ["location_x", "location_y", "dim", "patch"])


# TODO
#from gym.envs.registration import register
#
#register(
#    id='CrowdSim-v0',
#    entry_point='crowd_sim.envs:CrowdSim',
#)


class CrowdSimWrapper(CrowdSim):
        def generate_static_map_input(self, max_size, phase, config=None):
            """!
            Generates randomly located static obstacles (boxes and walls) in the environment.
                @param max_size: Max size in meters of the map
            """
            if config is not None:
                num_circles = config.getint('general', 'num_circles')
                num_walls = config.getint('general', 'num_walls')
            else:
                num_circles = int(round(np.random.random() * self.num_circles))
                num_walls = int(round(np.random.random() * self.num_walls))

            grid_size = int(round(max_size / self.map_resolution))
            self.map = np.ones((grid_size, grid_size))
            max_locations = int(round(grid_size))
            obstacles = []
            self.obstacle_vertices = []
            if phase == 'test':
                inflation_rate_il = 1
            else:
                inflation_rate_il = 1.25

            # for circle_index in range(num_circles):
            #     while True:
            #         if config is not None:
            #             location_x = config.getfloat(
            #                 'x_locations_circles', str(circle_index))
            #             location_y = config.getfloat(
            #                 'y_locations_circles', str(circle_index))
            #             circle_radius = config.getfloat(
            #                 'circle_radius', str(circle_index))
            #         else:
            #             location_x = np.random.randint(
            #                 -max_locations / 2.0, max_locations / 2.0)
            #             location_y = np.random.randint(
            #                 -max_locations / 2.0, max_locations / 2.0)
            #             circle_radius = (np.random.random() + 0.5) * 0.7
            #         dim = (int(round(2 * circle_radius / self.map_resolution)),
            #             int(round(2 * circle_radius / self.map_resolution)))
            #         patch = np.zeros([dim[0], dim[1]])

            #         location_x_m = location_x * self.map_resolution
            #         location_y_m = location_y * self.map_resolution

            #         collide = False
            #         if norm(
            #             (location_x_m - self.robot.px,
            #             location_y_m - self.robot.py)) < circle_radius + self.robot.radius + self.discomfort_dist or norm(
            #             (location_x_m - self.robot.gx,
            #             location_y_m - self.robot.gy)) < circle_radius + self.robot.radius + self.discomfort_dist:
            #             collide = True
            #         if not collide:
            #             break
            #     obstacles.append(Obstacle(int(round(location_x + grid_size / 2.0)),
            #                             int(round(location_y + grid_size / 2.0)), dim, patch))
            #     circle_radius_inflated = inflation_rate_il * circle_radius
            #     self.obstacle_vertices.append([(location_x_m +
            #                                     circle_radius_inflated, location_y_m +
            #                                     circle_radius_inflated), (location_x_m -
            #                                                             circle_radius_inflated, location_y_m +
            #                                                             circle_radius_inflated), (location_x_m -
            #                                                                                         circle_radius_inflated, location_y_m -
            #                                                                                         circle_radius_inflated), (location_x_m +
            #                                                                                                                 circle_radius_inflated, location_y_m -
            #                                                                                                                 circle_radius_inflated)])

            # for wall_index in range(num_walls):
            #     while True:
            #         if config is not None:
            #             location_x = config.getfloat(
            #                 'x_locations_walls', str(wall_index))
            #             location_y = config.getfloat(
            #                 'y_locations_walls', str(wall_index))
            #             x_dim = config.getfloat('x_dim', str(wall_index))
            #             y_dim = config.getfloat('y_dim', str(wall_index))
            #         else:
            #             location_x = np.random.randint(
            #                 -max_locations / 2.0, max_locations / 2.0)
            #             location_y = np.random.randint(
            #                 -max_locations / 2.0, max_locations / 2.0)
            #             if np.random.random() > 0.5:
            #                 x_dim = np.random.randint(2, 4)
            #                 y_dim = 1
            #             else:
            #                 y_dim = np.random.randint(2, 4)
            #                 x_dim = 1
            #         dim = (int(round(x_dim / self.map_resolution)),
            #             int(round(y_dim / self.map_resolution)))
            #         patch = np.zeros([dim[0], dim[1]])

            #         location_x_m = location_x * self.map_resolution
            #         location_y_m = location_y * self.map_resolution

            #         collide = False

            #         if (abs(location_x_m -
            #                 self.robot.px) < x_dim /
            #             2.0 +
            #             self.robot.radius +
            #             self.discomfort_dist and abs(location_y_m -
            #                                         self.robot.py) < y_dim /
            #             2.0 +
            #             self.robot.radius +
            #             self.discomfort_dist) or (abs(location_x_m -
            #                                         self.robot.gx) < x_dim /
            #                                     2.0 +
            #                                     self.robot.radius +
            #                                     self.discomfort_dist and abs(location_y_m -
            #                                                                 self.robot.gy) < y_dim /
            #                                     2.0 +
            #                                     self.robot.radius +
            #                                     self.discomfort_dist):
            #             collide = True
            #         if not collide:
            #             break

            # obstacles.append(Obstacle(int(round(location_x + grid_size / 2.0)),
            #                         int(round(location_y + grid_size / 2.0)), dim, patch))
            # x_dim_inflated = inflation_rate_il * x_dim
            # y_dim_inflated = inflation_rate_il * y_dim
            # self.obstacle_vertices.append([(location_x_m +
            #                                 x_dim_inflated /
            #                                 2.0, location_y_m +
            #                                 y_dim_inflated /
            #                                 2.0), (location_x_m -
            #                                     x_dim_inflated /
            #                                     2.0, location_y_m +
            #                                     y_dim_inflated /
            #                                     2.0), (location_x_m -
            #                                             x_dim_inflated /
            #                                             2.0, location_y_m -
            #                                             y_dim_inflated /
            #                                             2.0), (location_x_m +
            #                                                     x_dim_inflated /
            #                                                     2.0, location_y_m -
            #                                                     y_dim_inflated /
            #                                                     2.0)])

            # for obstacle in obstacles:
            #     if obstacle.location_x > obstacle.dim[0] / 2.0 and \
            #             obstacle.location_x < grid_size - obstacle.dim[0] / 2.0 and \
            #             obstacle.location_y > obstacle.dim[1] / 2.0 and \
            #             obstacle.location_y < grid_size - obstacle.dim[1] / 2.0:

            #         start_idx_x = int(
            #             round(
            #                 obstacle.location_x -
            #                 obstacle.dim[0] /
            #                 2.0))
            #         start_idx_y = int(
            #             round(
            #                 obstacle.location_y -
            #                 obstacle.dim[1] /
            #                 2.0))
            #         self.map[start_idx_x:start_idx_x +
            #                 obstacle.dim[0], start_idx_y:start_idx_y +
            #                 obstacle.dim[1]] = np.minimum(self.map[start_idx_x:start_idx_x +
            #                                                         obstacle.dim[0], start_idx_y:start_idx_y +
            #                                                         obstacle.dim[1]], obstacle.patch)

            #     else:
            #         for idx_x in range(obstacle.dim[0]):
            #             for idx_y in range(obstacle.dim[1]):
            #                 shifted_idx_x = idx_x - obstacle.dim[0] / 2.0
            #                 shifted_idx_y = idx_y - obstacle.dim[1] / 2.0
            #                 submap_x = int(
            #                     round(
            #                         obstacle.location_x +
            #                         shifted_idx_x))
            #                 submap_y = int(
            #                     round(
            #                         obstacle.location_y +
            #                         shifted_idx_y))
            #                 if submap_x > 0 and submap_x < grid_size and submap_y > 0 and submap_y < grid_size:
            #                     self.map[submap_x,
            #                             submap_y] = obstacle.patch[idx_x, idx_y]

            if self.robot.policy.name != 'SDOADRL' and self.robot.policy.name != 'ORCA':
                self.create_observation_from_static_obstacles(obstacles)
    




class CurriculumEnv(NavRepTrainEnv):
    """Curiculum Env Wrapper"""
    def __init__(self, scenario='test', silent=False, legacy_mode=False, adaptive=True, lidar_legs=True, collect_statistics=True):
        super().__init__(scenario=scenario, silent=silent, legacy_mode=legacy_mode, adaptive=adaptive, lidar_legs=lidar_legs, collect_statistics=collect_statistics)

    def _make_env(self, silent=True):
        #return super()._make_env(silent=silent)
        # Create env
        config_dir = resource_filename('crowd_nav', 'config')
        config_file = os.path.join(config_dir, 'test_soadrl_static.config')
        config_file = os.path.expanduser(config_file)
        config = configparser.RawConfigParser()
        config.read(config_file)

        #TODO change
        #env = gym.make('CrowdSim-v0')
        env = CrowdSimWrapper()
        env.configure(config, silent=silent)
        robot = Robot(config, 'humans')
        env.set_robot(robot)

        policy = SDOADRLDummyPolicy()
        policy.configure(config)
        if self.LEGACY_MODE:
            sess = tf.Session()
            policy = SDOADRL()
            policy.configure(sess, 'global', config)
            policy.set_phase('test')
            policy.load_model(os.path.expanduser('~/soadrl/Final_models/angular_map_full_FOV/rl_model'))

        env.robot.set_policy(policy)
        if not silent:
            env.robot.print_info()

        self.soadrl_sim = env

if __name__=='__main__' :
    from navrep.tools.envplayer import EnvPlayer
    env = CurriculumEnv()
    player = EnvPlayer(env)
