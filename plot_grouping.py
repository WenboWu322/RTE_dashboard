import matplotlib.pyplot as plt
import numpy as np

# 示例数据
caissons = ['Caisson 1', 'Caisson 2', 'Caisson 3']
time_points = [
    [1, 2, 3, 4, 5],  # Caisson 1 的时间点
    [1.5, 2.5, 3.5, 4.5],  # Caisson 2 的时间点
    [2, 3, 4, 5]  # Caisson 3 的时间点
]
highlighted_points = [4, 3.5, 5]  # 高亮点

# 绘图
fig, ax = plt.subplots(figsize=(10, 4))

for i, (caisson, points) in enumerate(zip(caissons, time_points)):
    y = [i] * len(points)  # y 值固定，表示每个 Caisson 的位置
    ax.plot(points, y, 'o-', label=caisson)  # 绘制线和点
    ax.scatter(highlighted_points[i], i, color='orange', s=100)  # 高亮特定点

# 添加标签和样式
ax.set_yticks(range(len(caissons)))
ax.set_yticklabels(caissons)
ax.set_xlabel('Time')
ax.set_title('Timeline Visualization')
plt.legend()
plt.show()
