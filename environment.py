import numpy as np
import cv2


'''
从图片获取地图

path = ''

img = cv2.imread(path)
h, w, _ = img.shape
rate = 0.6
grid = cv2.resize(img, (int(h * rate), int(w * rate)))[:, :, 0] // 255
grid = np.array(grid)
grid[grid == 1] = 255

'''

'''
# 随机获取地图

h, w = 20, 20
rate = 0.1
start = (0, 0)
end = (5, 5)
size = h * w
vis = np.zeros(size, dtype='bool')
vis[start] = vis[end] = True
obstacles = np.random.choice(
    [_ for _ in range(size) if not vis[_]],
    size=int((size - 2) * rate),
    replace=False
)

x = obstacles // w
y = obstacles % w
grid[x, y] = 0   # color is black

'''
