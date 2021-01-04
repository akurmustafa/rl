import numpy as np

class Grid:
    def __init__(self, width, heigth, discount = 0.9):
        self.width = width
        self.heigth = heigth
        self.x_pos = 0
        self.y_pos = 0
        self.values = np.zeros((heigth, width))
        self.discount = discount
        self.vertex_sources = []
        self.vertex_dests = []
        self.vertex_values = []
    
    def init_rewards(self, rewards):
        assert rewards.shape[0] == self.heigth and rewards.shape[1]==self.width, "reward initialized is not valid"
        self.rewards = rewards
    
    def add_vertex(self, source, dest):
        assert len(source) == 2 and len(dest) == 2, "source or dest is not valid"
        self.vertex_sources.append(source)
        self.vertex_dests.append(dest)

    def update(self):
        next_values = np.zeros((self.heigth, self.width))
        for x in range(self.width):
            for y in range(self.heigth):
                if [y, x] in self.vertex_sources:
                    for vertex_source, vertex_dest in zip(self.vertex_sources, self.vertex_dests):
                        if [y, x] == vertex_source:
                            next_values[y, x] += self.rewards[y,x] + self.discount*self.values[vertex_dest[0], vertex_dest[1]]
                            break
                else:
                    for cur_movement, cur_prob in zip([[-1, 0], [0, 1], [1, 0], [0, -1]], [0.25, 0.25, 0.25, 0.25]):
                        next_place = [y+cur_movement[0], x+cur_movement[1]]
                        if 0<=next_place[0]<self.heigth and 0<=next_place[1]<self.width:
                            next_values[y, x] += cur_prob*(self.rewards[y,x] + self.discount*self.values[next_place[0], next_place[1]])
                        else:
                            next_values[y, x] += cur_prob*(-1+self.discount*self.values[y, x])
        print('-'*20)
        print (next_values)
        self.values = next_values

    def policy(self):
        movement_x = -1
        movement_y = -1
        if np.random.rand()> 0.5:
            movement_x = 1
        if np.random.rand()>0.5:
            movement_y = 1
        if [self.x_pos, self.y_pos] in self.vertex_sources:
            for vertex_source, vertex_dest in zip(self.vertex_sources, self.vertex_dests):
                if vertex_source == [self.x_pos, self.y_pos]:
                    self.x_pos = vertex_dest[0]
                    self.y_pos = vertex_dest[1]
        else:
            if 0<=self.x_pos+movement_x<self.heigth:
                self.x_pos+=movement_x
            

grid_world = Grid(5, 5, discount = 0.9)
reward = np.zeros((5, 5))
grid_world.add_vertex([0, 1], [4, 1])
reward[0, 1] = 10
grid_world.add_vertex([0, 3], [2, 3])
reward[0, 3] = 5
grid_world.init_rewards(reward)
print(grid_world.values)
for i in range(50):
    print('iter: {}'.format(i))
    grid_world.update()
