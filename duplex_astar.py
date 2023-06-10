from astar import AStar, IndexNode
import heapq
import time


class DuplexAStar(AStar):
    def __init__(self, static_map, seed=2):
        super().__init__(static_map, seed)

    def extensionNode(self, open_list, open_dict, closed_dict, curNode, goal):
        '''扩展节点'''
        cur = (curNode.x, curNode.y)
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
                nextNode = IndexNode(next[0], next[1], goal, curNode)
                heapq.heappush(open_list, nextNode)
                open_dict.update({next : nextNode})

        del open_dict[cur]
        closed_dict.update({cur : curNode})

        return (open_list[0].x, open_list[0].y)

    def get_path(self, forward_curNode, backward_curNode):
        self.route_length = round(forward_curNode.g + backward_curNode.g, 3)

        if forward_curNode.x == backward_curNode.x and \
            forward_curNode.y == backward_curNode.y:
            backward_curNode = backward_curNode.father

        while forward_curNode:
            x, y = forward_curNode.x, forward_curNode.y
            self.path_x.append(x)
            self.path_y.append(y)
            forward_curNode = forward_curNode.father
        
        self.path_x.reverse()
        self.path_y.reverse()

        while backward_curNode:
            x, y = backward_curNode.x, backward_curNode.y
            self.path_x.append(x)
            self.path_y.append(y)
            backward_curNode = backward_curNode.father

    def run(self):
        forward_open_list, forward_open_dict = [], {}
        forward_closed_dict = {}
        backward_open_list, backward_open_dict = [], {}
        backward_closed_dict = {}

        start = self.to2DIndex(self.start)
        end = self.to2DIndex(self.end)

        startNode = IndexNode(start[0], start[1], end)
        endNode = IndexNode(end[0], end[1], start)

        heapq.heapify(forward_open_list) # 堆化
        heapq.heappush(forward_open_list, startNode)
        forward_open_dict.update({start : startNode})

        heapq.heapify(backward_open_list) 
        heapq.heappush(backward_open_list, endNode)
        backward_open_dict.update({end : endNode})

        start_time = time.time()
        forward_goal = end
        backward_goal = start
    
        while forward_open_list and backward_open_list:
            forward_curNode = heapq.heappop(forward_open_list)
            backward_curNode = heapq.heappop(backward_open_list)

            forward_cur = (forward_curNode.x, forward_curNode.y)
            backward_cur = (backward_curNode.x, backward_curNode.y)

            if forward_cur == backward_cur:
                self.get_path(forward_curNode, backward_curNode)
                break
            elif forward_cur in backward_closed_dict:
                self.get_path(forward_curNode, backward_closed_dict[forward_cur])
                break
            elif backward_cur in forward_closed_dict:
                self.get_path(backward_curNode, forward_closed_dict[backward_cur])
                break

            t_backward_goal = self.extensionNode(forward_open_list, forward_open_dict, \
                forward_closed_dict, forward_curNode, forward_goal)
            forward_goal = self.extensionNode(backward_open_list, backward_open_dict, \
                backward_closed_dict, backward_curNode, backward_goal)
            backward_goal = t_backward_goal

        end_time = time.time()

        self.search_time = round(end_time - start_time, 3)
        self.node_num = len(forward_open_list) + len(forward_closed_dict) + \
            len(backward_open_list) + len(backward_closed_dict)
        # self.plot(start, end)
        return 

