def plot_adj_matrix(adj_matrix, img_path='adj_matrix.png'):
    import matplotlib.pyplot as plt

    # 创建一个颜色映射，值为1的格子显示为蓝色，值为0的格子显示为白色
    fig, ax = plt.subplots()
    cax = ax.matshow(adj_matrix, cmap='Blues')

    # 在每个格子上写上矩阵的值
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            c = adj_matrix[i, j]
            ax.text(j, i, str(c), va='center', ha='center')

    # 去掉坐标轴
    plt.savefig(img_path)
    # 显示图像
    plt.show()



def plot_directed_graph(adj_matrix, img_path='directed_graph.png'):
    import matplotlib.pyplot as plt
    import networkx as nx
    
    nodes = range(adj_matrix.shape[0])
    # 创建有向图
    DG = nx.DiGraph()

    # 添加节点
    DG.add_nodes_from(nodes)

    # 根据邻接矩阵添加有向边
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] > 0:
                    DG.add_edge(i, j, weight=adj_matrix[i, j], width=adj_matrix[i, j]*10)

    # 绘制有向图
    pos = nx.spring_layout(DG)  # 使用spring布局
    plt.figure(figsize=(6,6))
    nx.draw(DG, pos, with_labels=True, node_color='lightgreen', node_size=700, font_size=12, font_color='black', edge_color='gray', arrows=True)
    plt.savefig(img_path)
