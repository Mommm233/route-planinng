import numpy as np
import heapq


class Node():
    def __init__(self, x, y, goal, father=None):
        self.x = x
        self.y = y
        self.h = self.get_estimate(goal)
        if father:
            self.update(father)
        else:
            self.g = 0
            self.f = self.h = self.get_estimate(goal)
            self.father = father

    def __lt__(self, a):
        return self.f < a.f
    
    def get_estimate(self, goal):
        '''h(n)'''
        return np.math.sqrt((self.x - goal[0]) ** 2 + (self.y - goal[1]) ** 2)

    def get_ground(self, next_pos):
        '''g(n)'''
        return 1 if abs(self.x  - next_pos[0]) + abs(self.y - next_pos[1]) == 1 else 1.414
    
    def update(self, father):
        self.g = father.g + self.get_ground((father.x, father.y))
        self.f = self.g + self.h
        self.father = father


class AStar():
    def __init__(self, grid, seed=2):
        np.random.seed(seed)
        self.grid = grid
        self.height, self.width = grid.shape
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def cross_boundry(self, pos):
        x, y = pos
        return x < 0 or x >= self.height or y < 0 or y >= self.width
    
    def check_collision(self, cur_pos, next_pos):
        if self.grid[next_pos] == 0: return True
        if abs(cur_pos[0] - next_pos[0]) + abs(cur_pos[1] - next_pos[1]) != 2:
            return False
        diagonal1, diagonal2 = (cur_pos[0], next_pos[1]), (next_pos[0], cur_pos[1])
        if not self.cross_boundry(diagonal1) and not self.cross_boundry(diagonal2):
            return self.grid[diagonal1] == 0 and self.grid[diagonal2] == 0
        else: return False

    def restore(self, curNode):
        path = []
        while curNode:
            path.append([curNode.x, curNode.y])
            curNode = curNode.father
        path.reverse()
        return np.array(path)

    def get_path(self, start, end):
        open_list = []
        open_dict, closed_dict = {}, {(-1, -1)}

        startNode = Node(start[0], start[1], end)
        heapq.heapify(open_list) # 堆化
        heapq.heappush(open_list, startNode)
        open_dict.update({start : startNode})

        while open_list:
            curNode = heapq.heappop(open_list)
            cur_pos = (curNode.x, curNode.y)

            if cur_pos == end:
                return self.restore(curNode)
            
            for d in self.directions:
                next_pos = (cur_pos[0] + d[0], cur_pos[1] + d[1])
                if next_pos in closed_dict or \
                    self.cross_boundry(next_pos) or \
                        self.check_collision(cur_pos, next_pos):
                    continue    # 已搜索 越界 障碍 
                if next_pos in open_dict:
                    nextNode = open_dict[next_pos]
                    if nextNode.g > curNode.g + curNode.get_ground(next_pos):
                        nextNode.update(curNode)
                        heapq.heapify(open_list)
                else:
                    nextNode = Node(next_pos[0], next_pos[1], end, curNode)
                    heapq.heappush(open_list, nextNode)
                    open_dict.update({next_pos : nextNode})

            del open_dict[cur_pos]
            closed_dict.add(cur_pos)

        return np.array([])
    
