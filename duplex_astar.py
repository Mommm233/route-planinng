import numpy as np
import heapq
from astar import AStar, Node

class DuplexAStar(AStar):
    def __init__(self, grid, seed=2):
        super().__init__(grid, seed)

    def extensionNode(self, open_list, open_dict, closed_dict, curNode, goal):
        '''扩展节点'''
        cur_pos = (curNode.x, curNode.y)
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
                nextNode = Node(next_pos[0], next_pos[1], goal, curNode)
                heapq.heappush(open_list, nextNode)
                open_dict.update({next_pos : nextNode})

        del open_dict[cur_pos]
        closed_dict.update({cur_pos : curNode})

        return (open_list[0].x, open_list[0].y)

    def restore(self, forward_curNode, backward_curNode):
        path_x, path_y = [], []
        if forward_curNode.x == backward_curNode.x and \
            forward_curNode.y == backward_curNode.y:
            backward_curNode = backward_curNode.father

        while forward_curNode:
            path_x.append(forward_curNode.x)
            path_y.append(forward_curNode.y)
            forward_curNode = forward_curNode.father
        
        path_x.reverse()
        path_y.reverse()

        while backward_curNode:
            x, y = backward_curNode.x, backward_curNode.y
            path_x.append(x)
            path_y.append(y)
            backward_curNode = backward_curNode.father
        
        return np.column_stack([path_x, path_y])

    def get_path(self, start, end):
        forward_open_list, forward_open_dict = [], {}
        forward_closed_dict = {}
        backward_open_list, backward_open_dict = [], {}
        backward_closed_dict = {}

        startNode = Node(start[0], start[1], end)
        endNode = Node(end[0], end[1], start)

        heapq.heapify(forward_open_list) # 堆化
        heapq.heappush(forward_open_list, startNode)
        forward_open_dict.update({start : startNode})

        heapq.heapify(backward_open_list) 
        heapq.heappush(backward_open_list, endNode)
        backward_open_dict.update({end : endNode})

        forward_goal = end
        backward_goal = start
    
        while forward_open_list and backward_open_list:
            forward_curNode = heapq.heappop(forward_open_list)
            backward_curNode = heapq.heappop(backward_open_list)

            forward_cur = (forward_curNode.x, forward_curNode.y)
            backward_cur = (backward_curNode.x, backward_curNode.y)

            if forward_cur == backward_cur:
                return self.restore(forward_curNode, backward_curNode)
            elif forward_cur in backward_closed_dict:
                return self.restore(forward_curNode, backward_closed_dict[forward_cur])
            elif backward_cur in forward_closed_dict:
                return self.restore(backward_curNode, forward_closed_dict[backward_cur])

            t_backward_goal = self.extensionNode(forward_open_list, forward_open_dict, \
                forward_closed_dict, forward_curNode, forward_goal)
            forward_goal = self.extensionNode(backward_open_list, backward_open_dict, \
                backward_closed_dict, backward_curNode, backward_goal)
            backward_goal = t_backward_goal

        return np.array([])
