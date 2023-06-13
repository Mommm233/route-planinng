import numpy as np



class Qlearning():
    def __init__(self, grid, start, end, dist, seed=0):
        np.random.seed(seed)
        self.height, self.width = grid.shape
        self.grid = grid
        self.start = start
        self.end = end
        self.dist = dist
        self.actions = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)])
        self.q_table = np.zeros((self.height, self.width, len(self.actions)))


    def cross_boundry(self, pos):
        x, y = pos
        return x < 0 or x >= self.height or y < 0 or y >= self.width

    def check_collision(self, cur_state, next_state):
        if self.grid[next_state] == 0: return True
        if abs(cur_state[0] - next_state[0]) + abs(cur_state[1] - next_state[1]) != 2:
            return False
        diagonal1, diagonal2 = (cur_state[0], next_state[1]), (next_state[0], cur_state[1])
        if not self.cross_boundry(diagonal1) and not self.cross_boundry(diagonal2):
            return self.grid[diagonal1] == 0 and self.grid[diagonal2] == 0
        else: return False

    def get_next_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            return np.argmax(self.q_table[state])
        else:
            return np.random.randint(len(self.actions))

    def update_q_table(self, epochs=1000, alpha=0.1, gamma=0.9, epsilon=0.9, max_step=100):
        for epoch in range(epochs):
            cur_state = self.start
            done = False
            i = 0
            while not done and i < max_step:
                action = self.get_next_action(cur_state, epsilon)
                next_state = (cur_state[0] + self.actions[action][0], cur_state[1] + self.actions[action][1])
                if self.cross_boundry(next_state) or \
                    self.check_collision(cur_state, next_state):
                    r = -1
                    self.q_table[cur_state][action] += alpha * (r - self.q_table[cur_state][action])
                    done = True

                elif next_state == self.end:
                    r = 10 + 10 / (self.dist + 0.1)
                    self.q_table[cur_state][action] += alpha * (r + gamma * np.max(self.q_table[next_state]) - self.q_table[cur_state][action])
                    done = True
                else:
                    r = 0
                    self.q_table[cur_state][action] += alpha * (r + gamma * np.max(self.q_table[next_state]) - self.q_table[cur_state][action])
                    cur_state = next_state
                    i += 1

            # print(f'epoch:{epoch}, done:{done}')


    def get_path(self):
        self.update_q_table()
        path = [self.start]
        cur_state = self.start
        while cur_state != self.end:
            action = np.argmax(self.q_table[cur_state])
            cur_state = (cur_state[0] + self.actions[action][0], cur_state[1] + self.actions[action][1])
            path.append(cur_state)
        return path
