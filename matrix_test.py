# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from plot import plot_adj_matrix, plot_directed_graph

def generate_partition_matrix(n, k):
    """
    生成一个 n 行 k 列的分区矩阵 P，每行恰好有一个非零项（1），且列是正交的。
    确保每列至少有一个值为1。
    
    参数:
    n: 行数（节点数量）
    k: 列数（簇的数量）
    
    返回:
    一个 n x k 的分区矩阵 P
    """
    if k > n:
        raise ValueError("列数 k 不能大于行数 n")

    # 初始化全为0的矩阵
    P = np.zeros((n, k), dtype=int)
    
    # 确保每列至少有一个值为1
    rows = np.random.choice(n, k, replace=False)
    for i in range(k):
        P[rows[i], i] = 1
    
    # 对于剩余的行随机选择一个列位置设置为1
    for i in range(n):
        if not np.any(P[i]):
            P[i, np.random.choice(k)] = 1
    
    return P


# # 随机生成一个10x10的邻接矩阵，元素为0或1
# adj_matrix = np.random.randint(0, 2, size=(6, 6))

# # 去除自环，即对角线元素设为0
# np.fill_diagonal(adj_matrix, 0)

# p = generate_partition_matrix(6,4)

# # 调用函数绘制邻接矩阵
# plot_adj_matrix(adj_matrix, 'adj_matrix.png')
# plot_directed_graph(adj_matrix, 'directed_graph.png')

# W = p.T @ adj_matrix @ p
# print(p)
# # 调用函数绘制邻接矩阵
# plot_adj_matrix(W, 'adj_matrix_W.png')
# plot_directed_graph(W, 'directed_graph_W.png')




import networkx as nx
import string
import matplotlib.pyplot as plt
import numpy as np

# 生成一个图，节点数为n
n = 4  # 你可以改变n的值来生成不同数量的节点

# 创建一个带有n个节点的图
G = nx.DiGraph()

# 节点标签从A开始，按字母顺序排列
labels = list(string.ascii_uppercase[:n])

# 添加节点
G.add_nodes_from(labels)

# 确保没有孤立节点的随机添加边
for i in range(n):
    connected = False
    for j in range(n):
        if i != j and np.random.rand() > 0.5:  # 50%概率添加边
            G.add_edge(labels[i], labels[j])
            connected = True
    # 如果没有添加任何边，强制添加一条边
    if not connected:
        j = np.random.choice([x for x in range(n) if x != i])
        G.add_edge(labels[i], j)


# 绘制图形
pos = nx.spring_layout(G)
plt.figure(figsize=(6,6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=12, font_color='black', edge_color='gray', arrows=True)
plt.show()


#生成新图N，N是这样得到的：p=generate_partition_matrix(n,k)，p(i,j)=1就意味着G的第i个节点label要append到N的第j个节点的label上
N = nx.MultiDiGraph()
k = 3
p = generate_partition_matrix(n, k)
print(p)

# Create new labels for the coarsened graph
new_labels = []
for j in range(k):
    cluster_label = ''
    for i in range(n):
        if p[i, j] == 1:
            cluster_label += labels[i]
    new_labels.append(cluster_label)

# Add nodes to the new graph with the new labels
N.add_nodes_from(new_labels)

# Add edges to the new graph based on the original graph G and the partition matrix p
for i in range(n):
    for j in range(n):
        if G.has_edge(labels[i], labels[j]):
            src_cluster = np.argmax(p[i])
            dst_cluster = np.argmax(p[j])
            if src_cluster != dst_cluster:
                N.add_edge(new_labels[src_cluster], new_labels[dst_cluster])

# Draw the coarsened graph
pos = nx.spring_layout(N)
plt.figure(figsize=(6, 6))

nx.draw(N, pos, with_labels=True, node_color='lightgreen', node_size=700, font_size=12, font_color='black', edge_color='gray', arrows=True)
plt.show()