import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astar import AStar

directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]

HEIGHT, WIDTH = 10, 10



static_map = np.zeros((HEIGHT, WIDTH), dtype='int') + 255



class Qlearning(AStar):
    def __init__(self, static_map, num, fig, ax):
        AStar.__init__(self, static_map)

        self.fig = fig
        self.ax = ax

        self.dobx = np.random.randint(0, self.height, num)
        self.doby = np.random.randint(0, self.width, num)
        self.dobd = np.random.randint(0, 9, num)

        exist_index = np.logical_and.reduce((self.dobx * self.width + self.doby != self.start, \
                                       self.static_map[self.dobx, self.doby] != 0))
        self.dobx = self.dobx[exist_index]
        self.doby = self.doby[exist_index]
        self.dobd = self.dobd[exist_index]
        self.duplicate_removal(self.dobx, self.doby, self.dobd)
        # self.create_obstacles()
        # self.run()
        
        self.q_table = np.zeros((self.height * self.width, 8))
        
        for i in range(len(self.path_x) - 1):
            x = self.path_x[i]
            y = self.path_y[i]
            nx = self.path_x[i + 1]
            ny = self.path_y[i + 1]
            action = np.where(np.all(directions == (nx - x, ny - y), axis=1))[0][0]
            self.q_table[x * self.width + y][action] += 0.1 # A* reward


    def duplicate_removal(self, dobx, doby, dobd, dynamic_map=None):
        xy = np.column_stack((dobx, doby))
        _, indices, counts = np.unique(xy, return_index=True, return_counts=True, axis=0)
        repeated_indices = indices[counts > 1]
        common_indices = [np.where(np.all(xy == xy[i], axis=1))[0] for i in repeated_indices][0]

        n_removal = np.full(len(dobx), True)
        n_removal[common_indices] = False
        
        if dynamic_map:
            dynamic_map[dobx[common_indices], doby[common_indices]] = 0

        dobx = dobx[n_removal]
        doby = doby[n_removal]
        dobd = dobd[n_removal]
    
    def update(self, frame):
        pass

    def animation (self):
        ani = FuncAnimation(fig, self.update, frames=None, init_func=self.init, blit=True, interval=500)

    def update_table(self, cur, alpha=0.1, gamma=0.9, epsilon=0.9, epoch=100):
        old_dobx = self.dobx.copy()
        old_doby = self.doby.copy()
        old_dobd = self.dobd.copy()
        range_idx = np.where(((np.abs(old_dobx - cur2[0]) == 0) & (np.abs(old_doby - cur2[1]) == 1)) \
            | ((np.abs(old_doby - cur2[1]) == 0) & (np.abs(old_dobx - cur2[0]) == 1)) \
            | ((np.abs(old_dobx - cur2[0]) == 1) & (np.abs(old_doby - cur2[1]) == 1)))[0]
        old_dobx = old_dobx[range_idx]
        old_doby = old_doby[range_idx]
        old_dobd = old_dobd[range_idx]

        for e in range(epoch):
            step = 0

            dobx = old_dobx.copy()
            doby = old_doby.copy()
            dobd = old_dobd.copy()

            dynamic_map = np.full((self.height, self.width), False)
            dynamic_map[dobx, doby] = True

            while cur != self.end:
                q_table = self.q_table[cur, :]

                if np.random.uniform() > epsilon or np.all(q_table) == 0:
                    action = np.random.randint(8)
                else:
                    action = np.random.choice(np.where(q_table == np.max(q_table))[0])

                cur2 = self.to2DIndex(cur)
                next2 = (cur2[0] + self.directions[action][0], \
                         cur2[1] + self.directions[action][1])
                next = next2[0] * self.width + next2[1]

                if self.cross_boundry(next2) or self.check_obstacles(cur2, next2):
                    q_table[action] -= 1
                    continue

                if dynamic_map[next2]:
                    q_table[action] -= 1
                    continue
            
                if np.where(dobx + directions[dobd][0] == next2[0] != [] &\
                            doby + directions[dobd][1] == next2[1] != []):
                    q_table[action] -= 1
                    continue
                    
                dynamic_map[dobx, doby] = False



                dobx += directions[dobd][0]
                doby += directions[dobd][1]

                n_crossb = np.where(dobx >= 0 & dobx < self.height & doby >= 0 & doby < self.width)
                dobx = dobx[n_crossb]
                doby = doby[n_crossb]
                dobd = dobd[n_crossb]

                exist_index = np.where(self.static_map[dobx, doby] != 0)
                dobx = dobx[exist_index]
                doby = doby[exist_index]
                dobd = dobd[exist_index]

                self.duplicate_removal(dobx, doby, dobd, dynamic_map)

                if next == self.end:
                    r = 1
                else:
                    r = 0
                q_table[action] += alpha * (r + gamma * np.max(self.q_table[next, :]) - q_table[action])
                cur = next
                step += 1
            print(f'epoch:{e}, step:{step}')
        np.save('q_table', self.q_table)

        # start = self.to2DIndex(self.start)
        # x, y = [start[0]], [start[1]]
        # cur = self.start
        # while cur != self.end:
        #     action = np.argmax(self.q_table[cur, :])
        #     cur2 = self.to2DIndex(cur)
        #     next2 = (cur2[0] + self.directions[action][0], \
        #                 cur2[1] + self.directions[action][1])
        #     next = next2[0] * self.width + next2[1]
        #     x.append(next2[0])
        #     y.append(next2[1])
        #     cur = next
        # return x, y

    def run_ql(self):
        pass

fig, ax = plt.subplots()
ql = Qlearning(static_map,10,fig,ax)
# ql.init()
ql.animation()
plt.grid(True, color='black')
plt.show()

