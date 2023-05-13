import numpy as np
import matplotlib.pyplot as plt
import heapq
import time


class IndexNode():
    def __init__(self, x, y, end, father=None):
        self.x = x
        self.y = y
        self.h = self.get_estimate(end)
        if father:
            self.update(father)
        else:
            self.g = 0
            self.f = self.h = self.get_estimate(end)
            self.father = father

    def __lt__(self, a):
        return self.f < a.f
    
    def get_estimate(self, end):
        '''h(n)'''
        return np.math.sqrt((self.x - end[0]) ** 2 + (self.y - end[1]) ** 2)

    def get_ground(self, next):
        '''g(n)'''
        return 1 if abs(self.x  - next[0]) + abs(self.y - next[1]) == 1 else 1.414
    
    def update(self, father):
        self.g = father.g + self.get_ground((father.x, father.y))
        self.f = self.g + self.h
        self.father = father


class AStar():
    def __init__(self, static_map, seed=2):
        np.random.seed(seed)
        self.static_map = static_map
        self.height, self.width = static_map.shape

        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        

        size = self.height * self.width
        self.start = np.random.randint(size)
        self.end = np.random.randint(size)
        while self.start == self.end:
            self.end = np.random.randint(size)

        self.path_x = []
        self.path_y = []

        self.route_length = 0
        self.search_time = 0
        self.node_num = 0

    def to2DIndex(self, index):
        return (index // self.width, index % self.width)

    def create_obstacles(self, rate=0.1):
        size = self.height * self.width
        vis = np.zeros(size, dtype='bool')
        vis[self.start] = vis[self.end] = True
        obstacles = np.random.choice(
            [_ for _ in range(size) if not vis[_]],
            size=int((size - 2) * rate),
            replace=False
        )

        x, y = self.to2DIndex(obstacles)
        self.static_map[x, y] = 0   # color is black

    def cross_boundry(self, next):
        x, y = next
        return x < 0 or x >= self.height or y < 0 or y >= self.width
    
    def check_obstacles(self, cur, next):
        '''安全性的检查障碍物'''
        if self.static_map[next] == 0: return True
        if abs(cur[0] - next[0]) + abs(cur[1] - next[1]) != 2:
            return False
        # 斜线
        diagonal1, diagonal2 = (cur[0], next[1]), (next[0], cur[1])
        if not self.cross_boundry(diagonal1) and not self.cross_boundry(diagonal2):
            return self.static_map[diagonal1] == 0 and self.static_map[diagonal2] == 0
        else: return False

    def get_path(self, curNode):
        self.route_length = round(curNode.g, 3)
        while curNode:
            x, y = curNode.x, curNode.y
            self.path_x.append(x)
            self.path_y.append(y)
            curNode = curNode.father


    def plot(self, start, end):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)

        # axes[0].grid(True, color='black')
        ax.set_xticks(np.arange(-0.5, self.width, 1), np.arange(0, self.width + 1, 1), alpha=0)   # alpha=0隐藏刻度
        ax.set_yticks(np.arange(-0.5, self.height, 1), np.arange(self.height, -1, -1), alpha=0)

        ax.text(start[1], start[0], 'S', fontsize=20, fontweight='bold', color='black', ha='center', va='center')
        ax.text(end[1], end[0], 'E', fontsize=20, fontweight='bold', color='black', ha='center', va='center')

        ax.plot(self.path_y, self.path_x, 'b')
        ax.imshow(self.static_map, cmap='gray', vmin=0, vmax=255)
        

    def run(self):
        open_list = []
        open_dict, closed_dict = {}, {(-1, -1)}

        start = self.to2DIndex(self.start)
        end = self.to2DIndex(self.end)

        startNode = IndexNode(start[0], start[1], end)

        heapq.heapify(open_list) # 堆化
        heapq.heappush(open_list, startNode)
        open_dict.update({start : startNode})

        start_time = time.time()
        while open_list:
            curNode = heapq.heappop(open_list)
            cur = (curNode.x, curNode.y)

            if cur == end:
                self.get_path(curNode)
                break

            for d in self.directions:
                next = (cur[0] + d[0], cur[1] + d[1])
                if next in closed_dict or \
                    self.cross_boundry(next) or \
                        self.check_obstacles(cur, next):
                    continue    # 已搜索 越界 障碍 
                if next in open_dict:
                    nextNode = open_dict[next]
                    if nextNode.g > curNode.g + curNode.get_ground(next):
                        nextNode.update(curNode)
                        heapq.heapify(open_list)
                else:
                    nextNode = IndexNode(next[0], next[1], end, curNode)
                    heapq.heappush(open_list, nextNode)
                    open_dict.update({next : nextNode})

            del open_dict[cur]
            closed_dict.add(cur)
        end_time = time.time()

        self.search_time = round(end_time - start_time, 3)
        self.node_num = len(open_list) + len(closed_dict) - 1
        self.plot(start, end)
        return 
    
