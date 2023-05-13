from astar import AStar
import numpy as np
import matplotlib.pyplot as plt



directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

HEIGHT, WIDTH = 10, 10

static_map = np.zeros((HEIGHT, WIDTH), dtype='int') + 255

# fig, axes = plt.subplots(1, 2)



# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)





class Qlearning(AStar):
    def __init__(self, static_map):
        super().__init__(static_map)
        # self.create_obstacles()
        # self.run()
        
        self.q_table = np.zeros((self.height * self.width, 8))
        # self.vis_path = np.zeros((self.height * self.width), dtype='bool')
        
        # self.path_x = np.array(self.path_x)
        # self.path_y = np.array(self.path_y)
        # self.vis_path[self.path_x * self.width + self.path_y] = True

        # self.dynamic_map = np.zeros(static_map.shape, dtype='int') + 255


    # def create_obstacles_tod(self, rate=0.05):
    #     size = self.height * self.width
    #     vis = np.zeros(size, dtype='bool')
    #     # self.vis_path
    #     vis[self.start] = vis[self.end] = vis[self.vis_path]= True
    #     obstacles = np.random.choice(
    #         [_ for _ in range(size) if not vis[_]],
    #         size=int((size - 2) * rate),
    #         replace=False
    #     )

    #     x, y = self.to2DIndex(obstacles)
    #     self.dynamic_map[x, y] = 100 

    # def check_obstacles_toq(self, cur, next):
    #     # super().check_obstacles(cur, next)
    #     if self.static_map[next] == 0 or self.dynamic_map[next] == 100: return True
    #     if abs(cur[0] - next[0]) + abs(cur[1] - next[1]) != 2:
    #         return False
    #     # 斜线
    #     diagonal1, diagonal2 = (cur[0], next[1]), (next[0], cur[1])
    #     if not self.cross_boundry(diagonal1) and not self.cross_boundry(diagonal2):
    #         return (self.static_map[diagonal1] == 0 or self.dynamic_map[diagonal1] == 100) \
    #             and (self.static_map[diagonal2] == 0 or self.dynamic_map[diagonal2] == 100)
    #     else: return False

    def update_table(self, alpha=0.1, gamma=0.9, epsilon=0.9, epoch=100, max_episodes=100):
        # noise = np.random.rand(8)
        for e in range(epoch):
            cur = self.start
            # episodes = 0
            step = 0
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
                if next == self.end:
                    r = 1
                else:
                    r = 0
                q_table[action] += alpha * (r + gamma * np.max(self.q_table[next, :]) - q_table[action])
                cur = next
                step += 1
            print(f'epoch:{e}, step:{step}')
        np.save('q_table', self.q_table)
        start = self.to2DIndex(self.start)
        x, y = [start[0]], [start[1]]
        cur = self.start
        while cur != self.end:
            action = np.argmax(self.q_table[cur, :])
            cur2 = self.to2DIndex(cur)
            next2 = (cur2[0] + self.directions[action][0], \
                        cur2[1] + self.directions[action][1])
            next = next2[0] * self.width + next2[1]
            x.append(next2[0])
            y.append(next2[1])
            cur = next
        return x, y

ql = Qlearning(static_map)
ql.create_obstacles()
# ql.create_obstacles_tod()
x, y = ql.update_table()

ql.run()

fig = plt.figure()

plt.grid(True, color='black')
plt.xticks(np.arange(-0.5, ql.width, 1), np.arange(0, ql.width + 1, 1), alpha=0)   # alpha=0隐藏刻度
plt.yticks(np.arange(-0.5, ql.height, 1), np.arange(ql.height, -1, -1), alpha=0)
start = ql.to2DIndex(ql.start)
end = ql.to2DIndex(ql.end)
plt.text(start[1],start[0], 'S', fontsize=20, fontweight='bold', color='black', ha='center', va='center')
plt.text(end[1], end[0], 'E', fontsize=20, fontweight='bold', color='black', ha='center', va='center')
plt.imshow(ql.static_map, cmap='gray', vmin=0, vmax=100)
plt.plot(y, x, 'b')

plt.show()
# fig, ax = plt.subplots()
# ax.imshow(ql.dynamic_map)
# plt.show()


