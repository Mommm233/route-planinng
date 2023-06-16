import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from qlearning import Qlearning

'''
全局路径规划与局部路径规划结合
模拟运动轨迹，生成动画图
'''

class dynamic_obstacle():
    def __init__(self, x, y):
        self.x = x
        self.y = y

def get_dir_vec(start, end, step_size):
    direction = end - start  # 计算射线的方向向量
    distance = np.linalg.norm(direction)  # 计算射线长度
    direction /= distance  # 将方向向量归一化
    num_steps = int(distance / step_size)  # 计算射线需要前进的步数
    return num_steps, direction

def get_line_and_cur(start, end, obstacles, d):
    step_size = 0.4  # 步长参数，控制射线前进的距离
    num_steps, direction = get_dir_vec(start, end, step_size)
    line = [start]
    for i in range(1, num_steps):
        current_pos = start + i * step_size * direction  # 计算当前射线位置
        if check_collision_dynamic_obstacle(current_pos, obstacles, d):
            return line, tuple(start + (i - 1) * step_size * direction)
        line.append(current_pos)

    return line, None

def check_collision(start, end, grid):
    step_size = 0.4  # 步长参数，控制射线前进的距离
    num_steps, direction = get_dir_vec(start, end, step_size)

    for i in range(num_steps):
        current_pos = start + i * step_size * direction  # 计算当前射线位置
        current_pos = np.round(current_pos).astype('int')
        if grid[current_pos[0], current_pos[1]] == 0:
            return True # 碰撞
    return False  # 没有碰撞，返回False

def get_new_control_points(control_points, grid):
    start = control_points[0]
    new_control_points = [start]
    for i in range(len(control_points) - 1):
        end = control_points[i + 1]
        if check_collision(start, end, grid):
            start = control_points[i]
            new_control_points.append(start)

    new_control_points.append(control_points[-1])
    # print(control_points)
    # print(new_control_points)
    return np.array(new_control_points)

def check_collision_dynamic_obstacle(pos, dynamic_obstacles, d):
    x, y = pos
    for obstacle in dynamic_obstacles:
        x1, y1 = obstacle.x, obstacle.y
        if math.sqrt((x - x1) ** 2 + (y - y1) ** 2) <= d:
            return True
    return False

def get_trajectory(control_points, dynamic_obstacles, grid):
    trajectory = []     # 轨迹
    size = len(control_points) - 1
    i = 0
    while i < size:
        _cur = control_points[i]
        _goal = control_points[i + 1]
        line, _new_cur = get_line_and_cur(_cur , _goal, dynamic_obstacles, 0.7)
        if _new_cur != None:
            new_grid = grid.copy()
            for ob in dynamic_obstacles:
                if check_collision_dynamic_obstacle(_new_cur, [ob], 2):
                    new_grid[ob.x, ob.y] = 0
            int_new_cur = (int(round(_new_cur[0])), int(round(_new_cur[1])))
            int_goal = (int(round(_goal[0])), int(round(_goal[1])))
            dist = ((control_points[-1][0] - _goal[0]) ** 2) + \
                ((control_points[-1][1] - _goal[1]) ** 2)
            dist = math.sqrt(dist)
            path = Qlearning(new_grid, int_new_cur, int_goal, dist).get_path()
            
            path = [_new_cur] + path
            path = np.array(path)

            path = get_new_control_points(path, new_grid)

            control_points = np.concatenate([control_points[:i + 1,], path[:-1,], control_points[i + 1:,]])
            size = len(control_points) - 1

        i += 1

    for i in range(len(control_points) - 1):
        _cur = control_points[i]
        _goal = control_points[i + 1]
        line, _ = get_line_and_cur(_cur , _goal, dynamic_obstacles, 0.7)
        if trajectory == []:
            for i in range(len(line)):
                trajectory.append(line[0:i+1])
        else:
            tl = trajectory[-1]
            for i in range(len(line)):
                trajectory.append(tl + line[0:i+1])

    return trajectory

def animation(frame, ax, control_points, trajectory, grid, dynamic_obstacles):
    tr = np.array(trajectory[frame])
    ax.clear()
    ax.grid(True)
    ax.set_xticks(np.arange(-0.5, 20, 1), np.arange(0, 21, 1), alpha=0)
    ax.set_yticks(np.arange(-0.5, 20, 1), np.arange(20, -1, -1), alpha=0)
    ax.imshow(grid, cmap='gray', vmin=0, vmax=255)
    for ob in dynamic_obstacles:
        ax.plot([ob.y], [ob.x], 'x')
    ax.plot(control_points[:, 1], control_points[:, 0], 'b--')
    ax.text(control_points[0][1], control_points[0][0], 'S', fontsize=20, fontweight='bold', color='black', ha='center', va='center')
    ax.text(control_points[-1][1], control_points[-1][0], 'E', fontsize=20, fontweight='bold', color='black', ha='center', va='center')
    ax.plot(tr[:, 1], tr[:, 0], 'r')
    ax.plot(tr[:, 1][-1], tr[:, 0][-1], 'o')
    ax.scatter(tr[:, 1][-1], tr[:, 0][-1], s=3200, marker='o', facecolors='none', edgecolors='black')

if __name__ == "__main__":
    # 原始曲线的坐标
    x = [15, 16, 17, 17, 17, 16, 16, 17, 16, 15, 14, 13, 12, 12, 11, 10, 9, 8, 7, 6, 5, 5, 4, 3, 2]
    y = [7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 14, 13, 12, 11, 12, 12, 12, 12, 13, 14, 15, 16, 17]

    grid = np.array([255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255])
    grid = grid.reshape((20, 20))

    control_points = np.column_stack((x, y))
    control_points = control_points.astype('float64')
    control_points = get_new_control_points(control_points, grid)
    
    dynamic_obstacles = [dynamic_obstacle(3, 15), dynamic_obstacle(11, 14), dynamic_obstacle(11, 15)]

    trajectory = get_trajectory(control_points, dynamic_obstacles, grid)

    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, 
                        animation, 
                        frames=len(trajectory), 
                        fargs=(ax, control_points, trajectory, grid, dynamic_obstacles), 
                        interval=180)
    ani.save('display.gif', writer='pillow', fps=5)
    plt.show()
