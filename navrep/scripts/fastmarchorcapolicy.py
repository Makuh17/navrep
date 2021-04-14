
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.state import JointState
class FastmarchORCAPolicy(object):
    def __init__(self, suicide_if_stuck=False):
        self.simulator = ORCA()
        self.suicide_if_stuck = suicide_if_stuck

    def reset(self):
        self.simulator.reset()

    def predict(self, obs, env):
        self.simulator.time_step = env._get_dt()
        other_agent_states = [
            agent.get_observable_state() for agent in env.soadrl_sim.humans + env.soadrl_sim.other_robots]
        joint_state = JointState(env.soadrl_sim.robot.get_full_state(), other_agent_states),
        )  # bigger than any possible distance in the map
#       IF reset has been called
            # create map here
            # skeleton for map creation
#              self.map = CMap2D()
            # origin is the x y coordinates of the grid cell ij=(0, 0)
#              self.map.origin[0] = msg.info.origin.position.x
#              self.map.origin[1] = msg.info.origin.position.y
#              self.map.set_resolution(msg.info.resolution)
            # set occupancy to 1 for occupied squares
#              self.map._occupancy = np.zeros((100, 100))
#              for each vertice in env.soadrl_sim.obstacle_vertices,
                    # draw polygon in the map
#                     self.map._occupancy[10:14, 8:10] = 1.
#              self.map.HUGE_ = 100 * np.prod( self.map._occupancy.shape )
            # field
    #         self.field = self.map.fastmarch()
        # robot position in field
#         joint_state.self_state.px
#         joint_state.self_state.py
        # 
#         ij = self.map.xy_to_ij(xy)
        # 
#         gridfmpath8, jumps = self.map.path_from_dijkstra_field(self.field, start, connectedness= 8)
#         virtual_goal = gridfmpath8[20]  # look forward in path
#         joint_state.self_state.gx = virtual_goal
        #
        action = self.simulator.predict(
            joint_state,
            env.soadrl_sim.obstacle_vertices,
            env.soadrl_sim.robot,
        )
        if self.suicide_if_stuck:
            if action.v < 0.1:
                return Suicide()
        vx = action.v * np.cos(action.r)
        vy = action.v * np.sin(action.r)
        return np.array([vx, vy, 0.1*(np.random.random()-0.5)])
