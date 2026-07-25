import numpy as np
import pygsp
import torch
from graph_coarsening.coarsening_utils import coarsen
from pygsp.graphs import Graph
from torch_geometric.utils import to_scipy_sparse_matrix


def convert_torch_geometric_graph_to_pygsp_graph(graph):
    adjacency = to_scipy_sparse_matrix(
        graph.edge_index,
        num_nodes=graph.num_nodes,
    )
    return pygsp.graphs.Graph(adjacency)


def one_hot(labels, class_count):
    return torch.nn.functional.one_hot(labels, num_classes=class_count).float()


def extract_components(graph: Graph):
    if graph.A.shape[0] != graph.A.shape[1]:
        raise ValueError("A square adjacency matrix is required.")
    if graph.is_directed():
        raise NotImplementedError("Directed graphs are not supported.")

    components = []
    visited = np.zeros(graph.A.shape[0], dtype=bool)
    while not visited.all():
        stack = {np.flatnonzero(~visited)[0]}
        component = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            component.append(node)
            visited[node] = True
            neighbors = graph.A[node, :].nonzero()[1]
            stack.update(neighbor for neighbor in neighbors if not visited[neighbor])

        component.sort()
        subgraph = graph.subgraph(component)
        subgraph.info = {"orig_idx": component}
        components.append(subgraph)
    return components


def coarsen_multiple_subgraphs(dataset, coarsening_ratio, coarsening_method):
    graph = convert_torch_geometric_graph_to_pygsp_graph(dataset)
    components = extract_components(graph)
    print(f"number of subgraphs: {len(components)}")
    components.sort(key=lambda item: len(item.info["orig_idx"]), reverse=True)

    coarsening_matrices = []
    coarsened_subgraphs = []
    for component in components:
        if component.N > 10:
            matrix, coarsened_graph, _, _ = coarsen(
                component,
                r=coarsening_ratio,
                method=coarsening_method,
            )
        else:
            matrix = None
            coarsened_graph = None
        coarsening_matrices.append(matrix)
        coarsened_subgraphs.append(coarsened_graph)

    return components, coarsening_matrices, coarsened_subgraphs


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = True
    return mask


def splits(data, num_classes, split_type):
    if split_type == "fixed":
        return data
    if split_type not in {"random", "few"}:
        raise ValueError(f"Invalid split type: {split_type!r}.")

    indices = []
    for class_index in range(num_classes):
        index = (data.y == class_index).nonzero().view(-1)
        indices.append(index[torch.randperm(index.size(0))])

    if split_type == "random":
        train_index = torch.cat([index[:20] for index in indices])
        val_index = torch.cat([index[20:50] for index in indices])
        test_index = torch.cat([index[50:] for index in indices])
    else:
        train_index = torch.cat([index[:5] for index in indices])
        val_index = torch.cat([index[5:10] for index in indices])
        test_index = torch.cat([index[10:] for index in indices])

    data.train_mask = index_to_mask(train_index, data.num_nodes)
    data.val_mask = index_to_mask(val_index, data.num_nodes)
    data.test_mask = index_to_mask(test_index, data.num_nodes)
    return data


def create_new_masks(matrix, train_labels, val_labels):
    train_projection = matrix.dot(train_labels.numpy())
    val_projection = matrix.dot(val_labels.numpy())

    new_train_mask = torch.from_numpy(train_projection.sum(axis=1) != 0)
    train_class_count = torch.from_numpy((train_projection > 0).sum(axis=1))
    new_train_mask[train_class_count > 1] = False

    new_val_mask = torch.from_numpy(val_projection.sum(axis=1) != 0)
    val_class_count = torch.from_numpy((val_projection > 0).sum(axis=1))
    new_val_mask[val_class_count > 1] = False
    return new_train_mask, new_val_mask


def update_coarsen_edges(graph, coarsening_state):
    adjacency = graph.W.tocoo()
    row = adjacency.row + coarsening_state["coarsen_node"]
    col = adjacency.col + coarsening_state["coarsen_node"]
    coarsening_state["coarsen_rows"].append(row)
    coarsening_state["coarsen_cols"].append(col)


def process_component(
    subgraph,
    features,
    labels,
    train_mask,
    val_mask,
    num_classes,
    matrix,
    coarsened_graph,
    coarsening_state,
):
    keep = subgraph.info["orig_idx"]
    subgraph_features = features[keep]
    subgraph_labels = labels[keep]
    subgraph_train_mask = train_mask[keep]
    subgraph_val_mask = val_mask[keep]

    if not (subgraph_train_mask.any() or subgraph_val_mask.any()):
        return

    if matrix is not None:
        train_labels = one_hot(subgraph_labels, num_classes)
        train_labels[~subgraph_train_mask] = 0
        val_labels = one_hot(subgraph_labels, num_classes)
        val_labels[~subgraph_val_mask] = 0
        new_train_mask, new_val_mask = create_new_masks(
            matrix, train_labels, val_labels
        )

        projected_features = torch.from_numpy(matrix.dot(subgraph_features.numpy())).to(
            dtype=features.dtype
        )
        projected_train_labels = torch.from_numpy(
            matrix.dot(train_labels.numpy())
        ).argmax(dim=1)
        projected_val_labels = torch.from_numpy(matrix.dot(val_labels.numpy())).argmax(
            dim=1
        )

        coarsening_state["features"].append(projected_features)
        coarsening_state["train_labels"].append(projected_train_labels)
        coarsening_state["train_masks"].append(new_train_mask)
        coarsening_state["val_labels"].append(projected_val_labels)
        coarsening_state["val_masks"].append(new_val_mask)
        update_coarsen_edges(coarsened_graph, coarsening_state)
        coarsening_state["coarsen_node"] += coarsened_graph.N
        return

    coarsening_state["features"].append(subgraph_features)
    coarsening_state["train_labels"].append(subgraph_labels)
    coarsening_state["train_masks"].append(subgraph_train_mask)
    coarsening_state["val_labels"].append(subgraph_labels)
    coarsening_state["val_masks"].append(subgraph_val_mask)
    update_coarsen_edges(subgraph, coarsening_state)
    coarsening_state["coarsen_node"] += subgraph.N


def load_and_coarsen(dataset, subgraphs, matrices, coarsened_subgraphs, split_type):
    num_classes = int(dataset.y.max().item()) + 1
    data = splits(dataset, num_classes, split_type)
    coarsening_state = {
        "coarsen_node": 0,
        "coarsen_rows": [],
        "coarsen_cols": [],
        "features": [],
        "train_labels": [],
        "train_masks": [],
        "val_labels": [],
        "val_masks": [],
    }

    for subgraph, matrix, coarsened_graph in zip(
        subgraphs,
        matrices,
        coarsened_subgraphs,
        strict=True,
    ):
        process_component(
            subgraph,
            data.x,
            data.y,
            data.train_mask,
            data.val_mask,
            num_classes,
            matrix,
            coarsened_graph,
            coarsening_state,
        )

    if not coarsening_state["features"]:
        raise ValueError("No component contains training or validation nodes.")

    coarsen_features = torch.cat(coarsening_state["features"])
    coarsen_edge = torch.from_numpy(
        np.vstack(
            (
                np.concatenate(coarsening_state["coarsen_rows"]),
                np.concatenate(coarsening_state["coarsen_cols"]),
            )
        )
    ).long()
    print(f"coarsened feature shape: {tuple(coarsen_features.shape)}")

    return (
        data,
        coarsen_features,
        torch.cat(coarsening_state["train_labels"]).long(),
        torch.cat(coarsening_state["train_masks"]),
        torch.cat(coarsening_state["val_labels"]).long(),
        torch.cat(coarsening_state["val_masks"]),
        coarsen_edge,
    )
