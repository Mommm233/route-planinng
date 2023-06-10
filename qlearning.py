import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from duplex_astar import DuplexAStar
import time


class Qlearning():
    def __init__(self, static_map, route_path, seed=0):
        np.random.seed(seed)
        self.height, self.width = static_map.shape
        self.static_map = static_map
        self.start = tuple(route_path[0])
        self.end = tuple(route_path[-1])
        self.route_path = route_path
        self.actions = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)])
        self.q_table = np.zeros((self.height, self.width, len(self.actions)))
        self.path_x, self.path_y = [self.start[0]], [self.start[1]]
        self.rdob = np.array([[]])
        for i in range(len(route_path) - 1):
            x, y = route_path[i]
            nx, ny = route_path[i + 1]
            action = np.where(np.all(self.actions == (nx - x, ny - y), axis=1))[0][0]
            self.q_table[x, y, action] += 1 # A* reward

        self.fig, self.ax = plt.subplots()



    def cross_boundry(self, pos):
        x, y = pos
        return x < 0 or x >= self.height or y < 0 or y >= self.width

    def check_static_obstacle(self, cur, next):
        if self.static_map[next] == 0: return True
        if abs(cur[0] - next[0]) + abs(cur[1] - next[1]) != 2:
            return False
        diagonal1, diagonal2 = (cur[0], next[1]), (next[0], cur[1])
        if not self.cross_boundry(diagonal1) and not self.cross_boundry(diagonal2):
            return self.static_map[diagonal1] == 0 and self.static_map[diagonal2] == 0
        else: return False

    def check_dynamic_obstacle(self, cur, pos, r, type=0):
        x, y = cur
        pos_x, pos_y = pos[:, 0], pos[:, 1]
        abs_diff_x = np.abs(pos_x - x)
        abs_diff_y = np.abs(pos_y - y)
        range_ob = np.logical_and(abs_diff_x <= r, abs_diff_y <= r)
        if type == 0:
            return pos[range_ob]
        return pos[~range_ob]
    
    def create_dynamic_obstacles(self, rate=0.035):
        num = int(self.height * self.width * rate)
        pos_x = np.random.randint(0, self.height, num)
        pos_y = np.random.randint(0, self.width, num)
        self.dob_pos = np.column_stack((pos_x, pos_y))
        self.dob_pos = np.concatenate([self.dob_pos, np.array([[5, 0], [4, 0], [4, 1], [3, 1], [3, 2], [2, 2], [2, 3], [8, 2]])])
        self.dob_pos = self.dob_pos[self.static_map[self.dob_pos[:, 0], self.dob_pos[:, 1]] != 0]
        self.dob_pos = self.check_dynamic_obstacle(self.start, self.dob_pos, r=1, type=1)

    def get_next_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            return np.argmax(self.q_table[state])
        else:
            return np.random.randint(len(self.actions))

    def update_q_table(self, _cur, _pos, epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.9):
        a = time.time()
        for epoch in range(epochs):
            cur = _cur
            while cur != self.end:
                action = self.get_next_action(cur, epsilon)
                next = (cur[0] + self.actions[action][0], cur[1] + self.actions[action][1])

                if self.cross_boundry(next):
                    r = -1
                    self.q_table[cur][action] += alpha * (r - self.q_table[cur][action])
                    continue
                if self.check_static_obstacle(cur, next):
                    r = -1
                    self.q_table[cur][action] += alpha * (r + gamma * np.max(self.q_table[next]) - self.q_table[cur][action])
                    continue

                range_pos = self.check_dynamic_obstacle(next, _pos, r=0)
                if len(range_pos) > 0:
                    r = -10
                    self.q_table[cur][action] += alpha * (r + gamma * np.max(self.q_table[next]) - self.q_table[cur][action])
                    continue
                if next == self.end:
                    r = 10
                    self.q_table[cur][action] += alpha * (r + gamma * np.max(self.q_table[next]) - self.q_table[cur][action])
                    break

                r = 0
                self.q_table[cur][action] += alpha * (r + gamma * np.max(self.q_table[next]) - self.q_table[cur][action])

                cur = next

        b = time.time()
        # print(f'cur:{_cur}, time:{b - a}s')



    def run(self):
        cur = (self.path_x[-1], self.path_y[-1])
        if cur != self.end:
            rdob = self.check_dynamic_obstacle(cur, self.dob_pos, r=3)
            if rdob.size == 0 or \
                (len(rdob) == len(self.rdob) and \
                np.any(rdob == self.rdob)):
                pass
            else:
                self.update_q_table(cur, rdob)
            self.rdob = rdob

            action = np.argmax(self.q_table[cur[0], cur[1], :])
 
            cur = (cur[0] + self.actions[action][0], cur[1] + self.actions[action][1])

            self.path_x.append(cur[0])
            self.path_y.append(cur[1])

        # np.save('q_table', self.q_table)

    def update_animation(self, frame):
        self.run()
        self.ax.clear()
        self.ax.grid(True, color='black')
        self.ax.set_xticks(np.arange(-0.5, self.width, 1), np.arange(0, self.width + 1, 1), alpha=0)
        self.ax.set_yticks(np.arange(-0.5, self.height, 1), np.arange(self.height, -1, -1), alpha=0)

        self.ax.text(self.start[1], self.start[0], 'S', fontsize=20, fontweight='bold', color='black', ha='center', va='center')
        self.ax.text(self.end[1], self.end[0], 'E', fontsize=20, fontweight='bold', color='black', ha='center', va='center')
        self.ax.scatter(self.dob_pos[:, 1], self.dob_pos[:, 0], marker='o', s=60, color='red')
        self.ax.plot(self.route_path[:, 1], self.route_path[:, 0], linestyle='--')
        self.ax.scatter(self.path_y[-1], self.path_x[-1], marker='o', s=60, color='blue')
        # self.ax.scatter(self.path_y[-1], self.path_x[-1], marker='o', s=2400, facecolors='none', edgecolors='r')
        self.ax.scatter(self.path_y[-1], self.path_x[-1], marker='o', s=15000, facecolors='none', edgecolors='r')
        self.ax.plot(self.path_y, self.path_x, 'b')

        self.ax.imshow(self.static_map, cmap='gray', vmin=0, vmax=255)


    def run_animation(self):
        self.create_dynamic_obstacles()
        ani = FuncAnimation(self.fig, self.update_animation, frames=None, interval=120)
        plt.show()


# static_map = np.full((15, 15), 255)

# dast = DuplexAStar(static_map)
# dast.create_obstacles()
# dast.run()

# route_path = np.column_stack((dast.path_x, dast.path_y))
# ql = Qlearning(dast.static_map, route_path)
# ql.run_animation()
