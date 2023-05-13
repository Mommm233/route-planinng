import matplotlib.pyplot as plt
from environment import Getmap
from astar import AStar
from duplex_astar import DuplexAStar


path = './image/map.png'
static_map = Getmap(path).map

ast = AStar(static_map)
dast = DuplexAStar(static_map)

ast.run()
dast.run()
plt.show()

print(ast.route_length, ast.search_time, ast.node_num)

print(dast.route_length, dast.search_time, dast.node_num)


# t1 = 0
# t2 = 0

# for i in range(10):
#     ast = AStar(static_map)
#     ast.run()
#     t1 += ast.time

# for i in range(10):

#     dast = DuplexAStar(static_map)
#     dast.run()

#     t2 += dast.time


# print(t1 / 10)
# print(t2 / 10)