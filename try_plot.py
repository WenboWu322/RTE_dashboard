import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay

# 生成一些随机数据
np.random.seed(42)
X = np.random.rand(100, 2)  # 100个二维数据点

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Delaunay三角剖分
triang = Delaunay(X)

# 创建图形
plt.figure(figsize=(8, 6))

# 为每个三角形区域绘制颜色
for simplex in triang.simplices:
    # 获取三角形的三个顶点
    pts = X[simplex]
    # 聚类中心的平均位置
    centroid = np.mean(pts, axis=0)
    # 为每个三角形填充聚类颜色
    plt.fill(pts[:, 0], pts[:, 1], color=plt.cm.viridis(kmeans.predict([centroid])[0] / 3), alpha=0.3)

# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)

plt.title("Clustered Points with Irregular 'Bubble' Shapes")
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()
