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
from navrep.envs.roomgen import Room

import crowd_sim  # adds CrowdSim-v0 to gym  # noqa
from crowd_sim.envs.crowd_sim import CrowdSim  # reference to env code  # noqa
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.network_om import SDOADRL
from crowd_sim.envs.utils.action import ActionXYRot
from crowd_sim.envs.utils.info import Collision, CollisionOtherAgent, ReachGoal
rng = np.random.default_rng(12345)
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
    def __init__(self):
        super(CrowdSimWrapper, self).__init__()
        self.room_width = 15
        self.room_height = 12
        self.min_room_side_len = 1.5
        self.num_corridors = 4
        self.corridor_width = 1.5
        self.subdivide_large_rooms = True
        self.subdivide_area_limits = 30
        self.wall_width = 0.15
        self.door_width = 0.9
    def generate_static_map_input(self, max_size, phase, config=None):
        """!
        Generates randomly located static obstacles (boxes and walls) in the environment.
            @param max_size: Max size in meters of the map
        """
        self.obstacle_vertices = []
        main_room = np.array([[self.room_width,self.room_height],[-self.room_width,self.room_height],[-self.room_width,-self.room_height],[self.room_width,-self.room_height]])/2
        main_room = Room.from_vert(main_room)   
        axis = rng.choice([0,1])
        rooms = []
        rooms.append(main_room)
        split = main_room.split_room(axis, self.corridor_width)
        rooms = [r for r in split if  min(r.dim[0],r.dim[1]) > self.min_room_side_len]

        # Creates corridors
        for i in range(self.num_corridors):
            axis = 1 - axis
            #corridor_w *= 0.9
            num = len(rooms)
            for i in range(num):
                room = rooms.pop(0)
                split = room.split_room(axis, self.corridor_width)
                split = [r for r in split if min(r.dim[0],r.dim[1]) > self.min_room_side_len]
                rooms+= split if split else [room]
            r_idx = np.argsort([r.get_area() for r in rooms])#np.arange(len(rooms))
            rooms = [rooms[i] for i in r_idx]

        if self.subdivide_large_rooms:
            big_rooms = [i for i,r in enumerate(rooms) if r.get_area() > self.subdivide_area_limits and r.corridor_sides]
            removed = []
            while big_rooms:
                for i, br_idx in enumerate(big_rooms):
                    room = rooms.pop(br_idx-i)
                    axis = np.argmax(room.dim)
                    if not any(axis == np.mod(room.get_corridor_sides(),2)):
                        axis = 1 - axis
                    for j in range(4):
                        split = room.split_room(axis, 0)
                        split_if = [min(r.dim[0],r.dim[1]) > self.min_room_side_len for r in split]
                        if all(split_if):
                            rooms+= split
                            break
                        if j==3:
                            removed += [room]
                big_rooms = [i for i,r in enumerate(rooms) if r.get_area() > self.subdivide_area_limits and r.corridor_sides]
            rooms += removed
        for room in rooms:
            # for polygon in room.get_polygons():
            #     self.obstacle_vertices.append(polygon)
            # alternative that for sure uses same 
            for polygon in room.get_polygons():
                polygon_list = []
                for point in polygon:
                    polygon_list.append((point[0],point[1]))
                self.obstacle_vertices.append(polygon_list)
        print(self.obstacle_vertices)
            # if self.robot.policy.name != 'SDOADRL' and self.robot.policy.name != 'ORCA':
            #     self.create_observation_from_static_obstacles(obstacles)
    


def temp_flatten(contours):
    n_total_vertices = len(np.array(contours).flatten())/2 + len(contours)
    n_total_vertices = sum( [ len(contour) for contour in contours])+ len(contours)
    print("total vert: ", n_total_vertices)
    print(np.array(contours))
    flat_contours = np.zeros((n_total_vertices, 3), dtype=np.float32)
    v = 0
    for idx, polygon in enumerate(contours):
        # add first vertex last to close polygon
        for vertex in polygon + polygon[:1]:
            flat_contours[v,:] = np.array([idx, vertex[0], vertex[1]])
            v += 1
    return flat_contours

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

    def _add_border_obstacle(self):
        return

    def reset(self):
        self.steps_since_reset = 0
        self.episode_reward = 0
        _, _ = self.soadrl_sim.reset(self.scenario, compute_local_map=False)
        random_rot = ActionXYRot(0, 0, 10.*(np.random.random()-0.5))
        self.soadrl_sim.step(random_rot, compute_local_map=False, border=self.border)
        if not self.LEGACY_MODE:
            self._add_border_obstacle()
        contours = self.soadrl_sim.obstacle_vertices
        #print("contours: ", contours)
        #self.flat_contours = flatten_contours(contours)
        self.flat_contours = temp_flatten(contours)
        #print("flat contours: ", self.flat_contours)
        self.distances_travelled_in_base_frame = np.zeros((len(self.soadrl_sim.humans), 3))
        obs = self._convert_obs()
        if self.LEGACY_MODE:
            state, local_map, reward, done, info = self.soadrl_sim.step(
                ActionXYRot(0, 0, 0), compute_local_map=True, border=self.border)
            obs = (state, local_map)
        return obs

if __name__=='__main__' :
    from navrep.tools.envplayer import EnvPlayer
    env = CurriculumEnv()
    player = EnvPlayer(env)
