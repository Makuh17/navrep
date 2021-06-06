import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Wedge, Polygon
# from crowd_sim.envs.crowd_sim import CrowdSim
# from crowd_sim.envs.utils.robot import Robot
# import os
# from crowd_nav.policy.network_om import SDOADRL
rng = np.random.default_rng(12345)

class Room(object):
    def __init__(self, h, w, c):
        self.update_hwc(h,w,c)
        self.corridor_sides = []
        self.door = []
        self.door_w = 0.9
        self.wall_thickness = 0.1
    def update_hwc(self, h, w, c):
        self.dim = np.array([w,h])
        self.c = c
        
    @classmethod
    def from_vert(cls, vert):
        min_x, max_x = np.min(vert[:,0]),np.max(vert[:,0])
        min_y, max_y = np.min(vert[:,1]),np.max(vert[:,1])
        h = max_y - min_y
        w = max_x - min_x
        c = np.array([max_x+min_x,max_y+min_y])/2
        return cls(h,w,c)
    
    def get_vert(self):
        return self.c + np.array([[self.dim[0], self.dim[1]],[-self.dim[0],self.dim[1]],[-self.dim[0],-self.dim[1]],[self.dim[0],-self.dim[1]]])/2
    
    def get_area(self):
        return self.dim[0]*self.dim[1]
    
    def add_door(self, side=None):
        if side == None:
            s = rng.choice(self.corridor_sides)
        else:
            s = side
        if s == 0 or s == 2:
            axis = 1
        elif s == 1 or s == 3:
            axis = 0
        
        if s == 0 or s == 3:
            fact = 1
        elif s == 1 or s == 2:
            fact = -1
        door_point = np.zeros([2,2])
        door_point[:,axis] = self.c[axis] + fact*self.dim[axis]/2
        p = rng.beta(1,1)*(self.dim[1-axis]-self.door_w-self.wall_thickness*2) - (self.dim[1-axis]-self.door_w-self.wall_thickness*2)/2
        door_point[:,1-axis] = (self.c[1-axis] + p) + np.array([-self.door_w,self.door_w])/2
        door_point = door_point if s < 2  else np.flip(door_point,0) # ensure that order is correct relative to positive direction
        
        other_doors = [d for d in self.door if d[1]==s]
        
        collision = False
        for door in other_doors:
            #check if current door clashes with existing door NOT WORKING
            if any(door_point[:,1-axis] > np.min(door[0][:,1-axis])) and any(door_point[:,1-axis] < np.max(door[0][:,1-axis])):
                collision = True
                print("collision")
        
        if not collision:
            self.door.append((door_point,s))
        
        #      0
        #   -------
        #   |     |
        # 1 |     | 3
        #   |     |
        #   -------
        #      2
        
    def get_polygons(self):
        if len(self.door) < 1:
            return [self.get_vert()]
        else:
            polygon_list = []
            rectangle_list = []
            door_add = np.array([[0,-1],[1,0],[0,1],[-1,0]])*self.wall_thickness
            vert_add = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*self.wall_thickness
            verts = self.get_vert()
            verts_prime = verts+vert_add

            doors = self.door
            
            # sorting
            door_avg = np.array([np.mean(d[0], axis=0) for d in doors])-self.c
            first_vert_ang = np.arctan2(verts[0][1]-self.c[1], verts[0][0]-self.c[0])
            #print("first_vert_ang: ", first_vert_ang)
            sorted_door_idx = np.argsort(np.mod(np.arctan2(door_avg[:,1],door_avg[:,0])-first_vert_ang+np.pi*2,np.pi*2))
            #print("sorted index: ", sorted_door_idx)

            # create polygon 1
            cur_door_idx = sorted_door_idx[0]
            cur_door = doors[cur_door_idx]
            last_door = doors[sorted_door_idx[-1]]
            
            vert_base = np.concatenate([verts[last_door[1]+1:],verts[:cur_door[1]+1]])
            vert_base_prime = np.concatenate([verts_prime[last_door[1]+1:],verts_prime[:cur_door[1]+1]])
            
            vert_base = np.concatenate([last_door[0][0].reshape((1,-1)), vert_base, cur_door[0][1].reshape((1,-1))])
            vert_base_prime = np.concatenate([(last_door[0][0]+ door_add[last_door[1]]).reshape((1,-1)),
                                             vert_base_prime, (cur_door[0][1]+ door_add[cur_door[1]]).reshape((1,-1))])
            for j in range(vert_base.shape[0]-1):
                outer = vert_base[j:j+2].copy()
                outer_prime = vert_base_prime[j:j+2].copy()
                axis = np.argwhere(outer[0]==outer[1])
                inner = outer.copy()
                inner[:,axis] = outer_prime[:,axis]
                inner = np.flip(inner, 0)
                rectangle_list.append(np.concatenate([outer, inner]))

            vert_base_prime = np.flip(vert_base_prime,0)
            
            polygon_list.append(np.concatenate([vert_base, vert_base_prime]).copy())

            for i in range(len(sorted_door_idx)-1):
                cur_door = doors[sorted_door_idx[i]]
                next_door = doors[sorted_door_idx[i+1]]
                vert_base = verts[cur_door[1]+1:next_door[1]+1]
                vert_base_prime = verts_prime[cur_door[1]+1:next_door[1]+1]
                
                vert_base = np.concatenate([cur_door[0][0].reshape((1,-1)), vert_base, next_door[0][1].reshape((1,-1))])
                vert_base_prime = np.concatenate([(cur_door[0][0]+ door_add[cur_door[1]]).reshape((1,-1)),
                                             vert_base_prime, (next_door[0][1]+ door_add[next_door[1]]).reshape((1,-1))])
                for j in range(vert_base.shape[0]-1):
                    outer = vert_base[j:j+2].copy()
                    outer_prime = vert_base_prime[j:j+2].copy()
                    axis = np.argwhere(outer[0]==outer[1])
                    inner = outer.copy()
                    inner[:,axis] = outer_prime[:,axis]
                    inner = np.flip(inner, 0)
                    rectangle_list.append(np.concatenate([outer, inner]))
                
                vert_base_prime = np.flip(vert_base_prime,0)
                
                polygon_list.append(np.concatenate([vert_base, vert_base_prime]).copy())
            
        return polygon_list, rectangle_list
    
    
    def set_corridor_sides(self, l):
        l = set(l)
        self.corridor_sides = list(l)
        
    def get_corridor_sides(self):
        return self.corridor_sides

    def split_room(self, axis, corridor_w):
        min_, max_ = np.min(self.get_vert()[:,axis]),np.max(self.get_vert()[:,axis])
        split = rng.random()*(max_-min_)+min_
        split = rng.beta(1,1)*(max_-min_-corridor_w)+min_+corridor_w/2
        room1 = self.get_vert().copy()
        result = []
        room1[room1[:,axis]==min_,axis] = split+corridor_w/2
        room1 = Room.from_vert(room1)
        if axis == 0:
            side1 = 1
            side2 = 3
        else:
            side1 = 2
            side2 = 0
        if corridor_w < 0.1:
            corridor_sides1 = self.get_corridor_sides()
            if side1 in corridor_sides1:
                corridor_sides1.remove(side1)
            corridor_sides2 = self.get_corridor_sides()
            if side2 in corridor_sides2:
                corridor_sides2.remove(side2)
        else:
            corridor_sides1 = self.get_corridor_sides() + [side1]
            corridor_sides2 = self.get_corridor_sides() + [side2]
        
        room1.set_corridor_sides(corridor_sides1)
        room1.add_door()
        if rng.choice([True,False]):
            room1.add_door()
        
        room2 = self.get_vert().copy()
        room2[room2[:,axis]==max_,axis] = split-corridor_w/2
        room2 = Room.from_vert(room2)
        room2.set_corridor_sides(corridor_sides2)
        room2.add_door()
        if rng.choice([True,False]):
            room2.add_door()
        return [room1, room2]


def split_room(room, axis, corridor_w):
    min_, max_ = np.min(room.get_vert()[:,axis]),np.max(room.get_vert()[:,axis])
    split = rng.random()*(max_-min_)+min_
    split = rng.beta(1,1)*(max_-min_-corridor_w)+min_+corridor_w/2
    room1 = room.get_vert().copy()
    result = []
    room1[room1[:,axis]==min_,axis] = split+corridor_w/2
    room1 = Room.from_vert(room1)
    if axis == 0:
        side1 = 1
        side2 = 3
    else:
        side1 = 2
        side2 = 0
    if corridor_w < 0.1:
        corridor_sides1 = room.get_corridor_sides()
        if side1 in corridor_sides1:
            corridor_sides1.remove(side1)
        corridor_sides2 = room.get_corridor_sides()
        if side2 in corridor_sides2:
            corridor_sides2.remove(side2)
    else:
        corridor_sides1 = room.get_corridor_sides() + [side1]
        corridor_sides2 = room.get_corridor_sides() + [side2]
    
    room1.set_corridor_sides(corridor_sides1)
    room1.add_door()
    if rng.choice([True,False]):
        room1.add_door()
    
    room2 = room.get_vert().copy()
    room2[room2[:,axis]==max_,axis] = split-corridor_w/2
    room2 = Room.from_vert(room2)
    room2.set_corridor_sides(corridor_sides2)
    room2.add_door()
    if rng.choice([True,False]):
        room2.add_door()
    return [room1, room2]

# class CrowdSimRoomGen(CrowdSim):
#     def __init__(self):
#         super(CrowdSimRoomGen, self).__init__()
#         self.room_width = 15
#         self.room_height = 12
#         self.min_room_side_len = 1.5
#         self.num_corridors = 4
#         self.corridor_width = 1.5
#         self.subdivide_large_rooms = True
#         self.subdivide_area_limits = 30
#         self.wall_width = 0.15
#         self.door_width = 0.9
    
#     def generate_static_map_input(self):
#         main_room = np.array([[self.room_width,self.room_height],[-self.room_width,self.room_height],[-self.room_width,-self.room_height],[self.room_width,-self.room_height]])/2
#         main_room = Room.from_vert(main_room)   
#         axis = rng.choice([0,1])
#         rooms = []
#         rooms.append(main_room)
#         split = main_room.split_room(axis, self.corridor_width)
#         rooms = [r for r in split if  min(r.dim[0],r.dim[1]) > self.min_room_side_len]

#         # Creates corridors
#         for i in range(self.num_corridors):
#             axis = 1 - axis
#             #corridor_w *= 0.9
#             num = len(rooms)
#             for i in range(num):
#                 room = rooms.pop(0)
#                 split = room.split_room(axis, self.corridor_width)
#                 split = [r for r in split if min(r.dim[0],r.dim[1]) > self.min_room_side_len]
#                 rooms+= split if split else [room]
#             r_idx = np.argsort([r.get_area() for r in rooms])#np.arange(len(rooms))
#             rooms = [rooms[i] for i in r_idx]

#         if self.subdivide_large_rooms:
#             big_rooms = [i for i,r in enumerate(rooms) if r.get_area() > self.subdivide_area_limits and r.corridor_sides]
#             removed = []
#             while big_rooms:
#                 for i, br_idx in enumerate(big_rooms):
#                     room = rooms.pop(br_idx-i)
#                     axis = np.argmax(room.dim)
#                     if not any(axis == np.mod(room.get_corridor_sides(),2)):
#                         axis = 1 - axis
#                     for j in range(4):
#                         split = room.split_room(axis, 0)
#                         split_if = [min(r.dim[0],r.dim[1]) > self.min_room_side_len for r in split]
#                         if all(split_if):
#                             rooms+= split
#                             break
#                         if j==3:
#                             removed += [room]
#                 big_rooms = [i for i,r in enumerate(rooms) if r.get_area() > self.subdivide_area_limits and r.corridor_sides]
#             rooms += removed
#         for room in rooms:
#             # for polygon in room.get_polygons():
#             #     self.obstacle_vertices.append(polygon)
#             # alternative that for sure uses same 
#             for polygon in room.get_polygons():
#                 polygon_list = []
#                 for point in polygon:
#                     polygon_list.append((point[0],point[1]))
#                 self.obstacle_vertices.append(polygon_list)
